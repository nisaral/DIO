"""
OpenAI-compatible reverse proxy powered by DIO scheduling.

Wraps one or more vLLM / SGLang / TGI / Ollama servers. Clients point their
OpenAI SDK ``base_url`` at DIO; DIO picks a backend and forwards the request.
"""

# NOTE: do NOT use ``from __future__ import annotations`` here — FastAPI needs
# live type objects (e.g. Request) to inject the ASGI request correctly.

import asyncio
import hmac
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from dio._version import __version__
from dio.backends import Backend, BackendPool, tgi_generate_to_openai_chat
from dio.config import DIOConfig, ablation_from_name
from dio.scheduler import AdmissionError, Scheduler

log = logging.getLogger("dio.gateway")


def _extract_prompt(body: Dict[str, Any]) -> str:
    if body.get("messages"):
        parts = []
        for m in body["messages"]:
            c = m.get("content", "")
            if isinstance(c, list):
                # multimodal: join text parts
                c = " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                )
            parts.append(f"{m.get('role', 'user')}: {c}")
        return "\n".join(parts)
    return str(body.get("prompt") or "")


def _ollama_options_to_openai(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map Ollama options (e.g. num_predict, temperature) to OpenAI-compatible params."""
    if not options or not isinstance(options, dict):
        return {}
    mapped: Dict[str, Any] = {}
    if "temperature" in options:
        mapped["temperature"] = float(options["temperature"])
    if "top_p" in options:
        mapped["top_p"] = float(options["top_p"])
    if "num_predict" in options:
        mapped["max_tokens"] = int(options["num_predict"])
    if "stop" in options:
        mapped["stop"] = options["stop"]
    if "presence_penalty" in options:
        mapped["presence_penalty"] = float(options["presence_penalty"])
    if "frequency_penalty" in options:
        mapped["frequency_penalty"] = float(options["frequency_penalty"])
    return mapped



def _ollama_done_reason(choices: List[Dict[str, Any]]) -> str:
    """Ollama's ``done_reason`` from an OpenAI ``finish_reason``."""
    if not choices:
        return "stop"
    finish = choices[0].get("finish_reason")
    if finish == "length":
        return "length"
    if finish == "tool_calls":
        return "tool_calls"
    return "stop"


def _ollama_tool_call_deltas(acc: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI tool-call fragments -> the single-object form Ollama clients read.

    OpenAI splits one call across chunks (name first, then the JSON arguments in
    pieces); Ollama hands back a finished object per call, so the gateway has to
    accumulate and re-emit.
    """
    out: List[Dict[str, Any]] = []
    for idx in sorted(acc):
        slot = acc[idx]
        if not slot.get("name") and not slot.get("arguments"):
            continue
        out.append(
            {
                "id": slot.get("id") or f"call_{idx}",
                "type": "function",
                "function": {
                    "name": slot.get("name", ""),
                    "arguments": slot.get("arguments", ""),
                },
            }
        )
    return out


def _ollama_request_to_openai(body: Dict[str, Any]) -> Dict[str, Any]:
    """Map Ollama request-level fields (not just ``options``) onto OpenAI params.

    Ollama clients can ask for structured output (``format``) or tools. Those were
    previously dropped on the floor, so a client asking for JSON got free-form
    text back with no error and no warning.
    """
    mapped: Dict[str, Any] = {}

    fmt = body.get("format")
    if isinstance(fmt, str) and fmt.lower() == "json":
        mapped["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        # Ollama accepts a bare JSON schema here; OpenAI wraps it in a named object.
        mapped["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "ollama_format", "schema": fmt},
        }
    elif fmt is not None:
        log.warning("Unsupported Ollama 'format' value %r: not forwarded", fmt)

    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        mapped["tools"] = tools
        if body.get("tool_choice") is not None:
            mapped["tool_choice"] = body["tool_choice"]

    if body.get("keep_alive") is not None:
        # Model residency is engine-local: DIO cannot honour it for the backend and
        # forwarding an Ollama-ism upstream would be meaningless.
        log.debug("Ignoring Ollama keep_alive=%r (engine-local setting)", body.get("keep_alive"))
    return mapped


def _openai_error(
    message: str,
    status_code: int,
    err_type: str = "invalid_request_error",
    code: Optional[str] = None,
) -> JSONResponse:
    """An OpenAI-shaped error body.

    SDKs read ``error.message`` / ``error.type``; returning FastAPI's
    ``{"detail": ...}`` for the framework's own 404/405/422 paths makes them
    surface an unhelpful "unknown error" to the caller.
    """
    err: Dict[str, Any] = {"message": message, "type": err_type}
    if code:
        err["code"] = code
    return JSONResponse(status_code=status_code, content={"error": err})


def _validate_openai_body(body: Any, path: str, max_tokens_cap: int = 0) -> Optional[str]:
    """Return a human-readable problem with the request, or None if it is sane.

    Cheap shape checks only: they exist so an obviously abusive or malformed
    request is rejected with a 400 instead of being forwarded to an engine that
    will bill GPU-seconds for it (``max_tokens: 10**9`` used to hold a worker
    until the client gave up).
    """
    if not isinstance(body, dict):
        return "body must be a JSON object"
    if path == "chat":
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return "'messages' must be a non-empty array"
    else:
        prompt = body.get("prompt")
        if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
            return "'prompt' is required"
    for key in ("max_tokens", "max_completion_tokens", "max_new_tokens"):
        value = body.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            # JSON `1e9` is a float, and saying "must be a positive integer"
            # about a positive number sends the caller after the wrong bug.
            return f"'{key}' must be an integer (got {value!r})"
        if value <= 0:
            return f"'{key}' must be positive (got {value})"
        if max_tokens_cap and value > max_tokens_cap:
            return f"'{key}'={value} exceeds the configured cap of {max_tokens_cap}"
    return None


def _estimate_tokens_heuristic(prompt: str, body: Dict[str, Any]) -> int:
    # Legacy byte heuristic (inflates MAPE when tokenizer differs).
    out = int(body.get("max_tokens") or body.get("max_completion_tokens") or 64)
    return max(1, len(prompt) // 4) + max(1, out)


#: Paths that mutate scheduler state or re-point traffic, and therefore must never
#: be reachable without credentials when an API key is configured.
def _is_admin_path(path: str) -> bool:
    """Exact /debug or a /debug/ subtree -- not any path merely starting with it."""
    return path == "/debug" or path.startswith("/debug/")


def _supplied_api_key(request: Request) -> Optional[str]:
    """Read a caller-supplied admin key from either supported header."""
    auth = request.headers.get("authorization") or ""
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    key = request.headers.get("x-dio-api-key")
    return key.strip() if key and key.strip() else None


def _ndjson_error(message: str):
    """Yield a single Ollama-style ndjson error line (stream is already 503)."""

    async def _gen():
        yield (json.dumps({"error": message, "done": True}) + "\n").encode("utf-8")

    return _gen()


class _TokenCounter:
    """Prefer HF tokenizer; fall back to ⌊|prompt|/4⌋ + max_tokens."""

    def __init__(self, name: Optional[str], enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self._tok = None
        self._mode = "heuristic"
        if enabled and name:
            try:
                from transformers import AutoTokenizer  # type: ignore

                self._tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                self._mode = "hf"
                log.info("Token counter: HF tokenizer %s", name)
            except Exception as e:
                log.warning(
                    "Token counter: HF tokenizer unavailable (%s); using heuristic", e
                )

    def prompt_tokens(self, prompt: str) -> int:
        """Count only the prompt (no completion budget).

        ``count`` deliberately returns prompt + max_tokens because that is the
        routing *feature*. Ollama clients want the real prompt length, so keep a
        separate accessor instead of reusing the feature.
        """
        if self._tok is not None:
            try:
                return max(1, len(self._tok.encode(prompt, add_special_tokens=False)))
            except Exception:
                pass
        return max(1, len(prompt) // 4)

    def count(self, prompt: str, body: Dict[str, Any]) -> int:
        out = int(body.get("max_tokens") or body.get("max_completion_tokens") or 64)
        out = max(1, out)
        if self._tok is not None:
            try:
                # encode without special tokens for feature size (matches e2e usage better)
                n_prompt = len(self._tok.encode(prompt, add_special_tokens=False))
                return max(1, int(n_prompt) + out)
            except Exception:
                pass
        return _estimate_tokens_heuristic(prompt, body)

    @property
    def mode(self) -> str:
        return self._mode


class DIOGateway:
    """
    High-level entry point.

    Example::

        from dio import DIOGateway, Backend

        gw = DIOGateway(backends=[
            Backend(id="gpu0", base_url="http://127.0.0.1:8000"),
            Backend(id="gpu1", base_url="http://127.0.0.1:8001", tier="large"),
        ])
        gw.run()  # blocks on uvicorn :8085
    """

    def __init__(
        self,
        backends: Optional[List[Backend]] = None,
        config: Optional[DIOConfig] = None,
        model_map: Optional[Dict[str, List[str]]] = None,
        **config_overrides: Any,
    ) -> None:
        self.config = config or DIOConfig(**config_overrides)
        cfg = self.config
        abl = ablation_from_name(cfg.ablation)
        if cfg.nlms_mode == "single":
            abl.single_timescale = True

        self.pool = BackendPool(backends or [])
        # Multi-model routing: model_name -> [backend_ids]
        self.model_map: Dict[str, List[str]] = model_map or {}
        self.scheduler = Scheduler(
            strategy=cfg.strategy,
            dual=(cfg.nlms_mode == "dual" and not abl.single_timescale),
            ablation=abl,
            slo_ms=cfg.slo_ms,
            admission_off=cfg.admission_off,
            admission_mode=cfg.admission_mode,
            admission_percentile=cfg.admission_percentile,
            recent_latency_window=cfg.recent_latency_window,
            batch_size=cfg.batch_size,
            tier_mismatch_ms=cfg.tier_mismatch_ms,
            cache_bonus_ms=cfg.cache_bonus_ms,
            affinity_cache_size=cfg.affinity_cache_size,
            vram_soft_mb=cfg.vram_soft_limit_mb,
            vram_hard_mb=cfg.vram_hard_limit_mb,
            kv_cache_cost_ms=cfg.kv_cache_cost_ms,
            engine_queue_cost_ms=cfg.engine_queue_cost_ms,
            engine_prefix_hit_bonus_ms=cfg.engine_prefix_hit_bonus_ms,
            use_engine_metrics=cfg.engine_metrics,
            mu_fast=cfg.mu_fast,
            mu_slow=cfg.mu_slow,
            mu_bias=cfg.mu_bias,
            blend=cfg.fast_slow_blend,
            initial_slope=cfg.initial_slope,
            initial_intercept=cfg.initial_intercept,
            static_slope=cfg.static_slope,
            static_intercept=cfg.static_intercept,
            health_interval_s=cfg.health_interval_s,
            decision_log_size=cfg.decision_log_size,
            pred_history_size=cfg.pred_history_size,
        )
        # Prefer model-id as tokenizer name when not set
        tok_name = cfg.tokenizer_name
        if cfg.use_tokenizer and not tok_name and backends:
            # try first backend model hint via env or leave None (heuristic until set)
            tok_name = None
        self.token_counter = _TokenCounter(tok_name, enabled=cfg.use_tokenizer)
        for b in self.pool.list():
            self.scheduler.register(
                b.id,
                tier=b.tier,
                total_vram_mb=b.total_vram_mb,
                free_vram_mb=b.free_vram_mb,
            )

        self.app = self._build_app()
        self._client: Optional[httpx.AsyncClient] = None
        self._metrics_task = None
        self._health_task = None

    def add_backend(self, backend: Backend) -> None:
        self.pool.add(backend)
        self.scheduler.register(
            backend.id,
            tier=backend.tier,
            total_vram_mb=backend.total_vram_mb,
            free_vram_mb=backend.free_vram_mb,
        )

    @staticmethod
    def _routing_headers(resp: Response) -> Dict[str, str]:
        """Forward DIO's routing annotations from an internal JSONResponse.

        Every non-streaming path sets both ``X-DIO-Backend`` and
        ``X-DIO-E2E-Ms``, and every streaming path sets ``X-DIO-Backend`` before
        the body starts. (``X-DIO-E2E-Ms`` cannot appear on a stream: the headers
        are flushed before the response finishes, and no HTTP trailer is sent.)
        The Ollama *non-streaming* translations were dropping both, so an Ollama
        client could not tell which engine served the request.
        """
        out: Dict[str, str] = {}
        for key in ("X-DIO-Backend", "X-DIO-E2E-Ms"):
            value = resp.headers.get(key)
            if value:
                out[key] = value
        return out

    def _build_app(self) -> FastAPI:
        app = FastAPI(
            title="DIO — Distributed Inference Orchestrator",
            description="Predictive NLMS router wrapping OpenAI-compatible engines (vLLM, etc.)",
            version=__version__,
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.middleware("http")
        async def _admin_guard(request: Request, call_next):
            """Gate the admin surface with a shared key when one is configured.

            ``/debug/*`` can reset counters, fake VRAM pressure and register new
            backend URLs, so an open admin port is a traffic-hijack vector rather
            than merely an information leak.
            """
            key = self.config.api_key
            if (
                key
                and self.config.protect_debug
                and _is_admin_path(request.url.path)
            ):
                supplied = _supplied_api_key(request)
                if not supplied or not hmac.compare_digest(supplied, key):
                    return JSONResponse(
                        status_code=401,
                        content={
                            "error": {
                                "message": "admin API key required",
                                "type": "dio_unauthorized",
                                "code": "unauthorized",
                            }
                        },
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            return await call_next(request)

        @app.on_event("startup")
        async def _startup() -> None:
            import asyncio

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.request_timeout_s),
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
                follow_redirects=True,
            )
            if self.config.engine_metrics:
                self._metrics_task = asyncio.create_task(self._metrics_loop())
            self._health_task = asyncio.create_task(self._health_loop())
            if not self.config.api_key and self.config.host not in (
                "127.0.0.1",
                "localhost",
                "::1",
            ):
                log.warning(
                    "Admin endpoints (/debug/*) are UNAUTHENTICATED and this process is "
                    "bound to %s. Set DIO_API_KEY and/or keep DIO behind an "
                    "authenticating reverse proxy.",
                    self.config.host,
                )
            log.info(
                "DIO gateway ready | strategy=%s mode=%s hybrid_metrics=%s backends=%s",
                self.config.strategy,
                self.config.nlms_mode,
                self.config.engine_metrics,
                [b.id for b in self.pool.list()],
            )

        @app.on_event("shutdown")
        async def _shutdown() -> None:
            for task in (self._metrics_task, getattr(self, "_health_task", None)):
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except (Exception, asyncio.CancelledError):
                        pass
            if self._client is not None:
                await self._client.aclose()
                self._client = None

        @app.get("/healthz")
        @app.get("/health")
        async def healthz():
            health = self.scheduler.workers_health()
            unhealthy = sorted(k for k, v in health.items() if not v)
            admission = self.scheduler.metrics()["admission"]
            # Backend liveness is not the only thing worth reporting: a fleet can
            # be "up" while goodput collapses. Callers that only care about
            # liveness keep using "status".
            return {
                # "degraded" so monitoring is not blind: previously this said "ok"
                # while every request was being rejected for lack of a backend.
                "status": "degraded" if unhealthy else "ok",
                "service": "dio",
                "backends": len(self.pool.backends),
                "backends_registered": len(health),
                "backends_unhealthy": unhealthy,
                "strategy": self.config.strategy,
                "engine_metrics": self.config.engine_metrics,
                "slo": {
                    "slo_ms": admission["slo_ms"],
                    "avg_e2e_ms": round(admission["avg_e2e_ms"], 1),
                    "goodput_fraction": round(admission["goodput_fraction"], 4),
                    "rejected_slo": admission["rejected_slo"],
                    "rejected_slo_suppressed": admission.get("rejected_slo_suppressed", 0),
                },
            }

        @app.get("/v1/models")
        async def list_models():
            # Aggregate models from configured model_map + backend probe
            data = []
            seen = set()
            for m in self.model_map.keys():
                if m not in seen:
                    seen.add(m)
                    data.append({
                        "id": m,
                        "object": "model",
                        "owned_by": "dio/config",
                    })
            client = self._http()
            for b in self.pool.list():
                try:
                    r = await client.get(
                        b.models_url(), timeout=3.0, headers=b.auth_headers()
                    )
                    if r.status_code == 200:
                        j = r.json()
                        for m in j.get("data") or []:
                            mid = m.get("id") or b.id
                            if mid not in seen:
                                seen.add(mid)
                                item = dict(m)
                                item["id"] = mid
                                item["owned_by"] = f"dio/{b.id}"
                                data.append(item)
                except Exception:
                    mid = b.model or b.id
                    if mid and mid not in seen:
                        seen.add(mid)
                        data.append(
                            {
                                "id": mid,
                                "object": "model",
                                "owned_by": f"dio/{b.id}",
                            }
                        )
            if not data:
                data = [{"id": "dio-default", "object": "model", "owned_by": "dio"}]
            return {"object": "list", "data": data}

        @app.api_route("/v1/chat/completions", methods=["POST"])
        async def chat_completions(body: Dict[str, Any] = Body(...)):
            # Body(...) avoids Request-injection quirks across FastAPI versions.
            problem = _validate_openai_body(body, "chat", self.config.max_tokens_cap)
            if problem:
                return _openai_error(problem, 400, "invalid_request_error", "invalid_body")
            tier = "small"
            # Multi-model routing: restrict backends to those serving the requested model
            model_hint = body.get("model") or ""
            allowed = self._resolve_model_backends(model_hint)
            if allowed is not None and not allowed:
                # model_map is configured and nothing serves this name. Without
                # this the empty allow-list reaches the scheduler and surfaces as
                # a 503 "no healthy backends", which reads like a capacity outage.
                return _openai_error(
                    f"The model '{model_hint}' does not exist.",
                    404,
                    "invalid_request_error",
                    "model_not_found",
                )
            if body.get("stream"):
                return await self._proxy_stream(body, path="chat", tier=tier, allowed_backends=allowed)
            return await self._proxy_json(body, path="chat", tier=tier, allowed_backends=allowed)

        @app.api_route("/v1/completions", methods=["POST"])
        async def completions(body: Dict[str, Any] = Body(...)):
            problem = _validate_openai_body(body, "completions", self.config.max_tokens_cap)
            if problem:
                return _openai_error(problem, 400, "invalid_request_error", "invalid_body")
            tier = "small"
            model_hint = body.get("model") or ""
            allowed = self._resolve_model_backends(model_hint)
            if allowed is not None and not allowed:
                return _openai_error(
                    f"The model '{model_hint}' does not exist.",
                    404,
                    "invalid_request_error",
                    "model_not_found",
                )
            if body.get("stream"):
                return await self._proxy_stream(body, path="completions", tier=tier, allowed_backends=allowed)
            return await self._proxy_json(body, path="completions", tier=tier, allowed_backends=allowed)

        # Ollama Native Adapter endpoints (/api/*)
        @app.get("/api/version")
        async def ollama_version():
            from dio import __version__
            return {"version": __version__}

        @app.get("/api/tags")
        async def ollama_tags():
            """Ollama-compatible model listing."""
            models_list = []
            seen = set()
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for m in self.model_map.keys():
                if m not in seen:
                    seen.add(m)
                    models_list.append({
                        "name": m,
                        "model": m,
                        "modified_at": now_iso,
                        "size": 4000000000,
                        "digest": f"sha256:{abs(hash(m)):012x}",
                        "details": {
                            "parent_model": "",
                            "format": "gguf",
                            "family": "llama",
                            "families": ["llama"],
                            "parameter_size": "8B",
                            "quantization_level": "Q4_K_M",
                        },
                    })
            for b in self.pool.list():
                m = b.model or b.id
                if m and m not in seen:
                    seen.add(m)
                    models_list.append({
                        "name": m,
                        "model": m,
                        "modified_at": now_iso,
                        "size": 4000000000,
                        "digest": f"sha256:{abs(hash(m)):012x}",
                        "details": {
                            "parent_model": "",
                            "format": "gguf",
                            "family": "llama",
                            "families": ["llama"],
                            "parameter_size": "8B",
                            "quantization_level": "Q4_K_M",
                        },
                    })
            if not models_list:
                models_list = [{
                    "name": "dio-default",
                    "model": "dio-default",
                    "modified_at": now_iso,
                    "size": 0,
                    "digest": "sha256:000000000000",
                    "details": {"format": "gguf", "family": "llama"},
                }]
            return {"models": models_list}

        @app.api_route("/api/show", methods=["POST"])
        async def ollama_show(body: Dict[str, Any] = Body(...)):
            model = body.get("model") or body.get("name") or "dio-default"
            return {
                "modelfile": f"# Modelfile for {model}\nFROM {model}\nPARAMETER temperature 0.7",
                "parameters": "temperature 0.7",
                "template": "{{ .Prompt }}",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "8B",
                    "quantization_level": "Q4_K_M",
                },
            }

        @app.api_route("/api/chat", methods=["POST"])
        async def ollama_chat(body: Dict[str, Any] = Body(...)):
            """Ollama /api/chat translation endpoint."""
            problem = _validate_openai_body(body, "chat", self.config.max_tokens_cap)
            if problem:
                return JSONResponse(status_code=400, content={"error": problem})
            tier = "small"
            model_name = body.get("model") or ""
            allowed = self._resolve_model_backends(model_name)
            if allowed is not None and not allowed:
                return JSONResponse(
                    status_code=404, content={"error": f"model '{model_name}' not found"}
                )

            openai_body: Dict[str, Any] = {
                "model": model_name,
                "messages": body.get("messages") or [],
            }
            if "options" in body:
                openai_body.update(_ollama_options_to_openai(body.get("options")))
            openai_body.update(_ollama_request_to_openai(body))

            stream = body.get("stream", True)
            if stream:
                openai_body["stream"] = True
                return await self._proxy_ollama_stream(
                    openai_body, path="chat", tier=tier, is_chat=True, allowed_backends=allowed
                )

            resp = await self._proxy_json(openai_body, path="chat", tier=tier, allowed_backends=allowed)
            if resp.status_code >= 400:
                return resp

            raw_body = resp.body.decode("utf-8") if hasattr(resp, "body") else "{}"
            data = json.loads(raw_body)
            choices = data.get("choices") or []
            msg = {"role": "assistant", "content": ""}
            if choices and "message" in choices[0]:
                msg = choices[0]["message"]

            e2e_ms = float(resp.headers.get("X-DIO-E2E-Ms", 10.0))
            usage = data.get("usage") or {}
            ollama_resp = {
                "model": model_name or "dio-default",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": msg,
                "done": True,
                # Real Ollama always reports why generation stopped; clients branch on it.
                "done_reason": _ollama_done_reason(choices),
                "total_duration": int(e2e_ms * 1_000_000),
                "load_duration": 1_000_000,
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "eval_count": usage.get("completion_tokens", 0),
            }
            return JSONResponse(ollama_resp, headers=self._routing_headers(resp))

        @app.api_route("/api/generate", methods=["POST"])
        async def ollama_generate(body: Dict[str, Any] = Body(...)):
            """Ollama /api/generate translation endpoint."""
            problem = _validate_openai_body(body, "completions", self.config.max_tokens_cap)
            if problem:
                return JSONResponse(status_code=400, content={"error": problem})
            tier = "small"
            model_name = body.get("model") or ""
            allowed = self._resolve_model_backends(model_name)
            if allowed is not None and not allowed:
                return JSONResponse(
                    status_code=404, content={"error": f"model '{model_name}' not found"}
                )

            openai_body: Dict[str, Any] = {
                "model": model_name,
                "prompt": body.get("prompt") or "",
            }
            if "options" in body:
                openai_body.update(_ollama_options_to_openai(body.get("options")))
            openai_body.update(_ollama_request_to_openai(body))

            stream = body.get("stream", True)
            if stream:
                openai_body["stream"] = True
                return await self._proxy_ollama_stream(
                    openai_body, path="completions", tier=tier, is_chat=False, allowed_backends=allowed
                )

            resp = await self._proxy_json(openai_body, path="completions", tier=tier, allowed_backends=allowed)
            if resp.status_code >= 400:
                return resp

            raw_body = resp.body.decode("utf-8") if hasattr(resp, "body") else "{}"
            data = json.loads(raw_body)
            choices = data.get("choices") or []
            text = choices[0].get("text", "") if choices else ""

            e2e_ms = float(resp.headers.get("X-DIO-E2E-Ms", 10.0))
            usage = data.get("usage") or {}
            ollama_resp = {
                "model": model_name or "dio-default",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "response": text,
                "done": True,
                "done_reason": _ollama_done_reason(choices),
                "total_duration": int(e2e_ms * 1_000_000),
                "load_duration": 1_000_000,
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "eval_count": usage.get("completion_tokens", 0),
            }
            return JSONResponse(ollama_resp, headers=self._routing_headers(resp))


        # Research / ops endpoints
        @app.get("/debug/metrics")
        async def metrics():
            return self.scheduler.metrics()

        @app.get("/debug/engine")
        async def engine_metrics():
            """Scraped vLLM-style /metrics snapshots (hybrid cost inputs)."""
            m = self.scheduler.metrics()
            return {
                "enabled": m.get("use_engine_metrics"),
                "interval_s": self.config.metrics_interval_s,
                "snapshots": m.get("engine_metrics") or {},
                "affinity": m.get("affinity") or {},
            }

        @app.get("/debug/admission")
        async def admission():
            return self.scheduler.metrics()["admission"]

        @app.get("/debug/affinity")
        async def affinity():
            return self.scheduler.metrics().get("affinity") or {}

        @app.get("/debug/predictions")
        async def predictions(limit: int = 1000):
            m = self.scheduler.metrics()["prediction"]
            samples = m.get("samples") or []
            return {**m, "samples": samples[-limit:]}

        @app.get("/debug/workers")
        async def workers():
            m = self.scheduler.metrics()
            return {
                "worker_count": len(m["workers"]),
                "workers": list(m["workers"].keys()),
                "strategy": m["strategy"],
                "detail": m["workers"],
                "engine": m.get("engine_metrics") or {},
            }

        @app.post("/debug/reset_stats")
        async def reset_stats():
            self.scheduler.reset_stats()
            return {"status": "ok"}

        @app.post("/debug/chaos/vram")
        async def chaos_vram(req: Request):
            body = await req.json()
            wid = body.get("worker_id")
            free = float(body.get("free_vram_mb", 1500))
            if not wid:
                raise HTTPException(400, "worker_id required")
            self.scheduler.set_vram(wid, free)
            if wid in self.pool.backends:
                self.pool.backends[wid].free_vram_mb = free
            return {"status": "ok", "worker_id": wid, "free_vram_mb": free}

        @app.post("/debug/backends")
        async def add_backend_http(req: Request):
            """Hot-register a backend: {id, base_url, tier?, total_vram_mb?}."""
            try:
                body = await req.json()
            except Exception:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "body must be JSON", "type": "dio_bad_request"}},
                )
            if not isinstance(body, dict) or not body.get("id") or not body.get("base_url"):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "body must contain 'id' and 'base_url'",
                            "type": "dio_bad_request",
                        }
                    },
                )
            try:
                b = Backend(
                    id=body["id"],
                    base_url=body["base_url"],
                    tier=body.get("tier", "small"),
                    model=body.get("model"),
                    total_vram_mb=float(body.get("total_vram_mb", 24000)),
                    free_vram_mb=float(body.get("free_vram_mb", body.get("total_vram_mb", 24000))),
                )
            except (ValueError, TypeError) as e:
                # Invalid base_url / VRAM: a 400 the caller can act on, not a 500.
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": str(e), "type": "dio_bad_request"}},
                )
            self.add_backend(b)
            return {"status": "ok", "id": b.id}

        # Framework-generated errors (404 route, 405 method, 422 body schema) would
        # otherwise reach an OpenAI client as {"detail": ...}, which its SDK cannot
        # turn into a useful exception. Normalise them to the OpenAI error envelope.
        @app.exception_handler(StarletteHTTPException)
        async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
            detail = exc.detail if isinstance(exc.detail, str) else "request failed"
            if request.url.path.startswith("/api/"):
                # Ollama clients expect {"error": "..."} with a 4xx/5xx status.
                return JSONResponse(status_code=exc.status_code, content={"error": detail})
            err_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
            return _openai_error(detail, exc.status_code, err_type)

        @app.exception_handler(RequestValidationError)
        async def _body_error(request: Request, exc: RequestValidationError) -> JSONResponse:
            first = (exc.errors() or [{}])[0]
            loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
            message = first.get("msg", "invalid request body")
            if loc:
                message = f"{loc}: {message}"
            # OpenAI answers an unparseable body with 400; FastAPI's 422 is a
            # framework detail that SDKs surface as an opaque failure.
            if request.url.path.startswith("/api/"):
                return JSONResponse(status_code=400, content={"error": message})
            return _openai_error(message, 400, "invalid_request_error", "invalid_body")

        return app

    def _feature_tokens(self, prompt: str, body: Dict[str, Any]) -> int:
        """Routing feature = prompt + completion budget, clamped.

        Only the *model input* is clamped; the forwarded request is untouched.
        """
        tokens = self.token_counter.count(prompt, body)
        cap = self.config.token_feature_cap
        return min(tokens, cap) if cap and cap > 0 else tokens

    def _http(self) -> httpx.AsyncClient:
        """Lazy client so TestClient / early requests work without waiting on startup."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.request_timeout_s),
                follow_redirects=True,
            )
        return self._client

    def _resolve_model_backends(self, model_name: str) -> Optional[List[str]]:
        """
        Multi-model routing: return backend IDs that serve ``model_name``.

        If no model_map is configured or model is a generic default, returns None
        (= all backends eligible, legacy behavior).
        If model_map is configured and model is not found, returns [] so the
        scheduler safely raises AdmissionError instead of misrouting.
        """
        if not self.model_map or not model_name or model_name in ("default", "dio-default"):
            return None
        # Exact match
        if model_name in self.model_map:
            return self.model_map[model_name]
        # Case-insensitive match on the model id or its last path segment.
        # NOTE: an unanchored substring test (``lower in key``) would route a
        # request for one model to a backend serving a *different* one, and
        # forward_chat overwrites the model with the backend's own, so the caller
        # would get a 200 from the wrong model. Fail closed instead.
        lower = model_name.lower().strip()
        if len(lower) < 3:
            return []
        for key, backends in self.model_map.items():
            k = key.lower()
            if lower == k or k.rsplit("/", 1)[-1] == lower:
                return backends
        # Prefix match on the last path segment: "mistral" -> "mistral-7b-instruct".
        for key, backends in self.model_map.items():
            tail = key.lower().rsplit("/", 1)[-1]
            if tail.startswith(lower) or lower.startswith(tail):
                return backends
        # Explicit model constraints exist, but no backend serves this model
        return []

    async def _metrics_loop(self) -> None:
        """
        Non-invasive hybrid telemetry (contribution A): poll each backend's
        Prometheus /metrics and feed KV-cache / queue / prefix-hit into the
        joint cost function. Zero engine source patches — HTTP GET only.
        """
        import asyncio

        from dio.engine_metrics import parse_prometheus_text, snapshot_from_metrics

        interval = max(0.25, float(self.config.metrics_interval_s))
        while True:
            try:
                client = self._http()
                for b in self.pool.list():
                    try:
                        r = await client.get(
                            b.metrics_url(), timeout=2.0, headers=b.auth_headers()
                        )
                        if r.status_code in (401, 403):
                            log.warning(
                                "metrics scrape %s returned %s - check Backend(api_key=...); "
                                "hybrid KV/queue cost terms are inactive",
                                b.id,
                                r.status_code,
                            )
                        if r.status_code >= 400:
                            continue
                        snap = snapshot_from_metrics(parse_prometheus_text(r.text))
                        if snap.ok:
                            self.scheduler.set_engine_metrics(b.id, snap)
                    except Exception as e:
                        log.debug("metrics scrape %s: %s", b.id, e)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("metrics loop error")
            await asyncio.sleep(interval)

    async def _health_loop(self) -> None:
        """Background health probes — mark dead engines unhealthy."""
        import asyncio

        while True:
            try:
                client = self._http()
                for b in self.pool.list():
                    ok = await self.pool.probe_health(client, b.id)
                    self.scheduler.set_healthy(b.id, ok)
                    if not ok:
                        log.warning("Backend unhealthy: %s (%s)", b.id, b.base_url)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("health loop error")
            await asyncio.sleep(max(2.0, self.config.health_interval_s))

    async def _proxy_json(
        self, body: Dict[str, Any], path: str, tier: str, allowed_backends: Optional[List[str]] = None
    ) -> Union[JSONResponse, Response]:
        prompt = _extract_prompt(body)
        tokens = self._feature_tokens(prompt, body)
        client = self._http()

        try:
            worker_id, decision = self.scheduler.pick(prompt, tier=tier, tokens=tokens, allowed_backends=allowed_backends)
        except AdmissionError as e:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": str(e),
                        "type": "dio_admission_rejected",
                        "code": "service_unavailable",
                    }
                },
                headers={"Retry-After": str(e.retry_after_sec)},
            )

        t0 = time.perf_counter()
        try:
            if path == "chat":
                resp = await self.pool.forward_chat(
                    client, worker_id, body, self.config.request_timeout_s
                )
            else:
                resp = await self.pool.forward_completions(
                    client, worker_id, body, self.config.request_timeout_s
                )
            # Fallback chat -> completions if the engine has no chat route
            if resp.status_code == 404 and path == "chat":
                comp_body = {
                    "model": body.get("model"),
                    "prompt": prompt,
                    "max_tokens": body.get("max_tokens", 64),
                    "temperature": body.get("temperature", 0.7),
                }
                resp = await self.pool.forward_completions(
                    client, worker_id, comp_body, self.config.request_timeout_s
                )
        except Exception as e:
            self.scheduler.release(worker_id)
            log.exception("Backend %s failed", worker_id)
            raise HTTPException(502, f"backend {worker_id} error: {e}") from e

        e2e_ms = (time.perf_counter() - t0) * 1000.0
        backend = self.pool.get(worker_id)
        # Backend-reported usage is kept for observability only. The learner is fed
        # ``tokens`` -- the exact feature ``estimate()`` used at prediction time --
        # because the NLMS gradient is err/tokens; feeding a different token count
        # evaluates the update at the wrong point and makes the slope drift.
        reported_tokens: Optional[int] = None
        try:
            data = resp.json()
            # Normalize TGI /generate -> OpenAI chat shape so clients stay OpenAI SDK
            if (
                resp.status_code < 400
                and backend.api_style == "tgi_generate"
                and path == "chat"
            ):
                data = tgi_generate_to_openai_chat(
                    data,
                    model=backend.model or body.get("model") or "tgi",
                    prompt_tokens=self.token_counter.prompt_tokens(prompt),
                )
            usage = data.get("usage") or {} if isinstance(data, dict) else {}
            if usage.get("total_tokens"):
                reported_tokens = int(usage["total_tokens"])
            elif usage.get("completion_tokens"):
                reported_tokens = max(1, len(prompt) // 4) + int(usage["completion_tokens"])
        except Exception:
            data = {"raw": resp.text}

        # A 5xx is a request-level failure, not proof the engine is dead: evicting
        # on the first one rejects every later request until the next health probe
        # (~5 s), which empties the pool entirely if it holds a single backend.
        if resp.status_code >= 500:
            if self.scheduler.note_failure(worker_id):
                log.warning(
                    "Backend %s returned %s - out of rotation after repeated failures",
                    worker_id,
                    resp.status_code,
                )

        # Success flag: a failed request must not train the learner nor count
        # toward goodput.
        self.scheduler.feedback(worker_id, e2e_ms, tokens, success=resp.status_code < 400)

        if resp.status_code >= 400:
            return JSONResponse(status_code=resp.status_code, content=data)

        # Annotate routing for observability
        if isinstance(data, dict):
            data.setdefault("dio", {})
            data["dio"] = {
                "backend_id": worker_id,
                "backend_url": backend.base_url,
                "api_style": backend.api_style,
                "e2e_ms": round(e2e_ms, 2),
                "tokens": tokens,
                "reported_tokens": reported_tokens,
                "slo_ms": self.config.slo_ms,
                # Measured here; the cost terms inside `decision` are predictions
                # and modelled penalties, so they need not add up to e2e_ms.
                "over_budget": e2e_ms > self.config.slo_ms,
                "decision": decision.as_dict(),
            }
        return JSONResponse(
            content=data,
            headers={
                "X-DIO-Backend": worker_id,
                "X-DIO-E2E-Ms": f"{e2e_ms:.1f}",
                # Backpressure signal: an agent that only sees 200s otherwise has
                # no way to notice that the pool is over its budget.
                "X-DIO-Budget-Ms": f"{self.config.slo_ms:.0f}",
                "X-DIO-Over-Budget": "1" if e2e_ms > self.config.slo_ms else "0",
            },
        )

    async def _relay_stream(
        self,
        body: Dict[str, Any],
        path: str,
        tier: str,
        allowed_backends: Optional[List[str]],
        media_type: str,
    ) -> Union[JSONResponse, StreamingResponse]:
        """
        Raw byte-passthrough streaming (OpenAI SSE).

        The upstream connection is opened eagerly so that a non-2xx engine response
        is surfaced to the client as a real error status instead of being flattened
        into ``200 OK`` with an error body inside a stream.
        """
        prompt = _extract_prompt(body)
        tokens = self._feature_tokens(prompt, body)
        client = self._http()

        try:
            worker_id, _decision = self.scheduler.pick(
                prompt, tier=tier, tokens=tokens, allowed_backends=allowed_backends
            )
        except AdmissionError as e:
            # ``except ... as e`` unbinds ``e`` once this block exits, but the
            # generator below only runs later, driven by Starlette -- so capture
            # the message and retry hint eagerly (and JSON-encode the payload).
            message = str(e)
            retry_after = str(e.retry_after_sec)

            async def err_gen():
                payload = {
                    "error": {
                        "message": message,
                        "type": "dio_admission_rejected",
                        "code": "service_unavailable",
                    }
                }
                yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                err_gen(),
                status_code=503,
                media_type=media_type,
                headers={"Retry-After": retry_after},
            )

        b = self.pool.get(worker_id)
        url = b.chat_url() if path == "chat" else b.completions_url()
        payload = dict(body)
        payload["stream"] = True
        if b.model:
            payload["model"] = b.model

        t0 = time.perf_counter()
        request = client.build_request(
            "POST",
            url,
            json=payload,
            # Auth + timeout must match the non-streaming path, otherwise a
            # token-gated engine 401s every stream while JSON succeeds.
            headers=b.auth_headers(),
            timeout=b.timeout_s or self.config.request_timeout_s,
        )
        try:
            upstream = await client.send(request, stream=True)
        except Exception as e:
            self.scheduler.release(worker_id)
            log.exception("Backend %s failed to open stream", worker_id)
            raise HTTPException(502, f"backend {worker_id} error: {e}") from e

        if upstream.status_code >= 400:
            raw = await upstream.aread()
            status = upstream.status_code
            await upstream.aclose()
            # The request never streamed, so close out the pending slot promptly.
            self.scheduler.feedback(
                worker_id, (time.perf_counter() - t0) * 1000.0, tokens, success=False
            )
            if status >= 500 and self.scheduler.note_failure(worker_id):
                log.warning(
                    "Backend %s returned %s - out of rotation after repeated failures",
                    worker_id,
                    status,
                )
            text = raw.decode("utf-8", errors="replace")
            try:
                content = json.loads(text)
            except Exception:
                content = {"error": {"message": text, "type": "backend_error"}}
            return JSONResponse(status_code=status, content=content)

        async def gen():
            stream_ok = False
            client_gone = False
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
                stream_ok = True
            except asyncio.CancelledError:
                client_gone = True
                raise
            except Exception as e:
                # 200 + chunked headers are already on the wire, so the only way to
                # tell the client something broke is an in-band error event.
                log.warning("Backend %s stream aborted: %s", worker_id, e)
                self.scheduler.note_failure(worker_id)
                payload = {
                    "error": {
                        "message": f"upstream stream failed: {e}",
                        "type": "dio_stream_error",
                    }
                }
                yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            finally:
                e2e_ms = (time.perf_counter() - t0) * 1000.0
                await upstream.aclose()
                if client_gone:
                    # The caller hung up; the engine may be blameless, so return the
                    # pending slot without recording a sample.
                    self.scheduler.release(worker_id)
                else:
                    self.scheduler.feedback(
                        worker_id, e2e_ms, tokens, success=stream_ok
                    )

        return StreamingResponse(
            gen(),
            media_type=media_type,
            headers={
                "X-DIO-Backend": worker_id,
                "Cache-Control": "no-cache",
                # Headers are flushed before the stream's latency is known, so a
                # stream can advertise the budget and the *prediction* only.
                "X-DIO-Budget-Ms": f"{self.config.slo_ms:.0f}",
                "X-DIO-Predicted-Ms": f"{_decision.total_ms:.1f}",
            },
        )

    async def _proxy_stream(
        self, body: Dict[str, Any], path: str, tier: str, allowed_backends: Optional[List[str]] = None
    ) -> Union[JSONResponse, StreamingResponse]:
        return await self._relay_stream(body, path, tier, allowed_backends, "text/event-stream")

    async def _proxy_ollama_stream(
        self,
        body: Dict[str, Any],
        path: str,
        tier: str,
        is_chat: bool,
        allowed_backends: Optional[List[str]] = None,
    ) -> Union[JSONResponse, StreamingResponse]:
        """
        Stream handler for Ollama native clients.

        Translates upstream OpenAI SSE chunks (data: {...}\n\n) into Ollama's
        newline-delimited JSON (ndjson) format on the fly.
        """
        model_name = body.get("model") or "dio-default"
        prompt = _extract_prompt(body)
        tokens = self._feature_tokens(prompt, body)
        prompt_only = self.token_counter.prompt_tokens(prompt)
        client = self._http()

        try:
            worker_id, _decision = self.scheduler.pick(
                prompt, tier=tier, tokens=tokens, allowed_backends=allowed_backends
            )
        except AdmissionError as e:
            message = str(e)
            retry_after = str(e.retry_after_sec)
            return StreamingResponse(
                _ndjson_error(message),
                status_code=503,
                media_type="application/x-ndjson",
                headers={"Retry-After": retry_after},
            )

        b = self.pool.get(worker_id)
        url = b.chat_url() if path == "chat" else b.completions_url()
        payload = dict(body)
        payload["stream"] = True
        if b.model:
            payload["model"] = b.model
        # Token parity: ask OpenAI-compatible engines for a trailing usage chunk
        # so prompt_eval_count/eval_count report the engine's real token counts
        # -- the same numbers the non-streaming path copies from ``usage`` --
        # instead of a chunk/character estimate. Engines that reject the unknown
        # field 400 the whole request, so the send below retries once without it.
        if b.api_style == "openai":
            payload.setdefault("stream_options", {"include_usage": True})

        t0 = time.perf_counter()
        request = client.build_request(
            "POST",
            url,
            json=payload,
            # Auth + timeout must match the non-streaming path, otherwise a
            # token-gated engine 401s every stream while JSON succeeds.
            headers=b.auth_headers(),
            timeout=b.timeout_s or self.config.request_timeout_s,
        )
        try:
            upstream = await client.send(request, stream=True)
            if upstream.status_code == 400 and "stream_options" in payload:
                # Some OpenAI-compatible builds (older llama.cpp) reject the
                # field outright; the request is cheap to rebuild and an
                # estimate still beats a hard failure.
                await upstream.aread()
                await upstream.aclose()
                payload.pop("stream_options")
                request = client.build_request(
                    "POST",
                    url,
                    json=payload,
                    headers=b.auth_headers(),
                    timeout=b.timeout_s or self.config.request_timeout_s,
                )
                upstream = await client.send(request, stream=True)
        except Exception as e:
            self.scheduler.release(worker_id)
            log.exception("Backend %s failed to open stream", worker_id)
            raise HTTPException(502, f"backend {worker_id} error: {e}") from e

        if upstream.status_code >= 400:
            raw = await upstream.aread()
            status = upstream.status_code
            await upstream.aclose()
            self.scheduler.feedback(
                worker_id, (time.perf_counter() - t0) * 1000.0, tokens, success=False
            )
            if status >= 500 and self.scheduler.note_failure(worker_id):
                log.warning(
                    "Backend %s returned %s - out of rotation after repeated failures",
                    worker_id,
                    status,
                )
            text = raw.decode("utf-8", errors="replace").strip()
            try:
                parsed = json.loads(text)
                message = (
                    (parsed.get("error") or {}).get("message")
                    if isinstance(parsed.get("error"), dict)
                    else parsed.get("error")
                ) or text
            except Exception:
                message = text or f"backend returned HTTP {status}"
            # Ollama clients expect ndjson on the wire, so keep the shape.
            return StreamingResponse(
                _ndjson_error(str(message)),
                status_code=status,
                media_type="application/x-ndjson",
            )

        async def gen():
            buffer = ""
            gen_tokens = 0
            # Engine-reported counts (None until/unless a usage chunk arrives).
            reported_prompt: Optional[int] = None
            reported_gen: Optional[int] = None
            # Accumulated tool calls (OpenAI streams them in fragments) and the
            # engine's own finish_reason, so a streamed `length` stop is not
            # reported to an Ollama client as `stop`.
            tool_calls: Dict[int, Dict[str, Any]] = {}
            last_finish: Optional[str] = None
            finished = False
            stream_ok = False
            client_gone = False
            try:
                async for chunk_bytes in upstream.aiter_bytes():
                    buffer += chunk_bytes.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            finished = True
                            break
                        try:
                            chunk = json.loads(data_str)
                            # The include_usage chunk has choices == [] and is the
                            # only place a streaming engine reports real tokens.
                            chunk_usage = chunk.get("usage")
                            if isinstance(chunk_usage, dict) and chunk_usage:
                                if chunk_usage.get("prompt_tokens") is not None:
                                    reported_prompt = int(chunk_usage["prompt_tokens"])
                                if chunk_usage.get("completion_tokens") is not None:
                                    reported_gen = int(chunk_usage["completion_tokens"])
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0]
                            if choice.get("finish_reason"):
                                last_finish = choice["finish_reason"]
                            if is_chat:
                                delta = choice.get("delta") or {}
                                delta_text = delta.get("content") or ""
                                if delta_text:
                                    gen_tokens += 1
                                    yield (
                                        json.dumps(
                                            {
                                                "model": model_name,
                                                "created_at": time.strftime(
                                                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                                ),
                                                "message": {
                                                    "role": "assistant",
                                                    "content": delta_text,
                                                },
                                                "done": False,
                                            }
                                        )
                                        + "\n"
                                    ).encode("utf-8")
                                for frag in delta.get("tool_calls") or []:
                                    idx = int(frag.get("index") or 0)
                                    slot = tool_calls.setdefault(idx, {"arguments": ""})
                                    fn = frag.get("function") or {}
                                    if frag.get("id"):
                                        slot["id"] = frag["id"]
                                    if fn.get("name"):
                                        slot["name"] = fn["name"]
                                    if fn.get("arguments"):
                                        slot["arguments"] = slot.get("arguments", "") + str(
                                            fn["arguments"]
                                        )
                                    gen_tokens += 1
                                    yield (
                                        json.dumps(
                                            {
                                                "model": model_name,
                                                "created_at": time.strftime(
                                                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                                ),
                                                "message": {
                                                    "role": "assistant",
                                                    "content": "",
                                                    "tool_calls": _ollama_tool_call_deltas(
                                                        tool_calls
                                                    ),
                                                },
                                                "done": False,
                                            }
                                        )
                                        + "\n"
                                    ).encode("utf-8")
                            else:
                                delta_text = choice.get("text") or ""
                                if delta_text:
                                    gen_tokens += 1
                                    yield (
                                        json.dumps(
                                            {
                                                "model": model_name,
                                                "created_at": time.strftime(
                                                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                                ),
                                                "response": delta_text,
                                                "done": False,
                                            }
                                        )
                                        + "\n"
                                    ).encode("utf-8")
                        except Exception:
                            continue
                    if finished:
                        break

                stream_ok = True
                e2e_ms = (time.perf_counter() - t0) * 1000.0
                total_duration_ns = int(e2e_ms * 1_000_000)
                final_chunk: Dict[str, Any] = {
                    "model": model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "done": True,
                    "done_reason": _ollama_done_reason([{"finish_reason": last_finish}]),
                    "total_duration": total_duration_ns,
                    "load_duration": 1_000_000,
                    # Prefer the engine's numbers; the estimates (prompt chars/4,
                    # one per delta chunk) are the fallback for engines that do
                    # not support stream_options.include_usage.
                    "prompt_eval_count": (
                        reported_prompt if reported_prompt is not None else prompt_only
                    ),
                    "eval_count": reported_gen if reported_gen is not None else gen_tokens,
                }
                if is_chat:
                    final_chunk["message"] = {"role": "assistant", "content": ""}
                    if tool_calls:
                        final_chunk["message"]["tool_calls"] = _ollama_tool_call_deltas(tool_calls)
                else:
                    final_chunk["response"] = ""
                yield (json.dumps(final_chunk) + "\n").encode("utf-8")
            except asyncio.CancelledError:
                client_gone = True
                raise
            except Exception as e:
                log.warning("Backend %s stream aborted: %s", worker_id, e)
                self.scheduler.note_failure(worker_id)
                yield (
                    json.dumps({"error": f"upstream stream failed: {e}", "done": True}) + "\n"
                ).encode("utf-8")
            finally:
                e2e_ms = (time.perf_counter() - t0) * 1000.0
                await upstream.aclose()
                if client_gone:
                    self.scheduler.release(worker_id)
                else:
                    # Same feature the prediction used -- see _proxy_json.
                    self.scheduler.feedback(
                        worker_id, e2e_ms, tokens, success=stream_ok
                    )

        return StreamingResponse(
            gen(),
            media_type="application/x-ndjson",
            headers={
                "X-DIO-Backend": worker_id,
                "Cache-Control": "no-cache",
                "X-DIO-Budget-Ms": f"{self.config.slo_ms:.0f}",
                "X-DIO-Predicted-Ms": f"{_decision.total_ms:.1f}",
            },
        )

    def run(self, host: Optional[str] = None, port: Optional[int] = None, **uvicorn_kwargs: Any) -> None:
        import uvicorn

        uvicorn.run(
            self.app,
            host=host or self.config.host,
            port=port or self.config.port,
            **uvicorn_kwargs,
        )
