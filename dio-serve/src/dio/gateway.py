"""
OpenAI-compatible reverse proxy powered by DIO scheduling.

Wraps one or more vLLM / SGLang / TGI / Ollama servers. Clients point their
OpenAI SDK ``base_url`` at DIO; DIO picks a backend and forwards the request.
"""

# NOTE: do NOT use ``from __future__ import annotations`` here — FastAPI needs
# live type objects (e.g. Request) to inject the ASGI request correctly.

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from dio.backends import Backend, BackendPool, tgi_generate_to_openai_chat
from dio.config import DIOConfig, ablation_from_name
from dio.scheduler import AdmissionError, Scheduler

log = logging.getLogger("dio.gateway")


def _extract_prompt(body: Dict[str, Any]) -> str:
    if "messages" in body and body["messages"]:
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



def _estimate_tokens_heuristic(prompt: str, body: Dict[str, Any]) -> int:
    # Legacy byte heuristic (inflates MAPE when tokenizer differs).
    out = int(body.get("max_tokens") or body.get("max_completion_tokens") or 64)
    return max(1, len(prompt) // 4) + max(1, out)


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

    def _build_app(self) -> FastAPI:
        app = FastAPI(
            title="DIO — Distributed Inference Orchestrator",
            description="Predictive NLMS router wrapping OpenAI-compatible engines (vLLM, etc.)",
            version="0.2.0",
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

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
            return {
                "status": "ok",
                "service": "dio",
                "backends": len(self.pool.backends),
                "strategy": self.config.strategy,
                "engine_metrics": self.config.engine_metrics,
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
            client = self._client or httpx.AsyncClient()
            for b in self.pool.list():
                try:
                    r = await client.get(b.models_url(), timeout=3.0)
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
            tier = "small"
            # Multi-model routing: restrict backends to those serving the requested model
            model_hint = body.get("model") or ""
            allowed = self._resolve_model_backends(model_hint)
            if body.get("stream"):
                return await self._proxy_stream(body, path="chat", tier=tier, allowed_backends=allowed)
            return await self._proxy_json(body, path="chat", tier=tier, allowed_backends=allowed)

        @app.api_route("/v1/completions", methods=["POST"])
        async def completions(body: Dict[str, Any] = Body(...)):
            tier = "small"
            model_hint = body.get("model") or ""
            allowed = self._resolve_model_backends(model_hint)
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
            tier = "small"
            model_name = body.get("model") or ""
            allowed = self._resolve_model_backends(model_name)

            openai_body: Dict[str, Any] = {
                "model": model_name,
                "messages": body.get("messages") or [],
            }
            if "options" in body:
                openai_body.update(_ollama_options_to_openai(body.get("options")))

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
                "total_duration": int(e2e_ms * 1_000_000),
                "load_duration": 1_000_000,
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "eval_count": usage.get("completion_tokens", 0),
            }
            return JSONResponse(ollama_resp)

        @app.api_route("/api/generate", methods=["POST"])
        async def ollama_generate(body: Dict[str, Any] = Body(...)):
            """Ollama /api/generate translation endpoint."""
            tier = "small"
            model_name = body.get("model") or ""
            allowed = self._resolve_model_backends(model_name)

            openai_body: Dict[str, Any] = {
                "model": model_name,
                "prompt": body.get("prompt") or "",
            }
            if "options" in body:
                openai_body.update(_ollama_options_to_openai(body.get("options")))

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
                "total_duration": int(e2e_ms * 1_000_000),
                "load_duration": 1_000_000,
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "eval_count": usage.get("completion_tokens", 0),
            }
            return JSONResponse(ollama_resp)


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
            body = await req.json()
            b = Backend(
                id=body["id"],
                base_url=body["base_url"],
                tier=body.get("tier", "small"),
                model=body.get("model"),
                total_vram_mb=float(body.get("total_vram_mb", 24000)),
                free_vram_mb=float(body.get("free_vram_mb", body.get("total_vram_mb", 24000))),
            )
            self.add_backend(b)
            return {"status": "ok", "id": b.id}

        return app

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
        # Case-insensitive / partial match (e.g. "llama3" matches "meta-llama/Llama-3.2-3B-Instruct")
        lower = model_name.lower()
        for key, backends in self.model_map.items():
            if lower in key.lower() or key.lower() in lower:
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
                        r = await client.get(b.metrics_url(), timeout=2.0)
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
        tokens = self.token_counter.count(prompt, body)
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
            # Fallback chat → completions if 404
            if resp.status_code == 404 and path == "chat":
                # convert messages → prompt
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
        usage_tokens = tokens
        backend = self.pool.get(worker_id)
        try:
            data = resp.json()
            # Normalize TGI /generate → OpenAI chat shape so clients stay OpenAI SDK
            if (
                resp.status_code < 400
                and backend.api_style == "tgi_generate"
                and path == "chat"
            ):
                data = tgi_generate_to_openai_chat(
                    data, model=backend.model or body.get("model") or "tgi"
                )
            usage = data.get("usage") or {} if isinstance(data, dict) else {}
            if usage.get("total_tokens"):
                usage_tokens = int(usage["total_tokens"])
            elif usage.get("completion_tokens"):
                usage_tokens = max(1, len(prompt) // 4) + int(usage["completion_tokens"])
        except Exception:
            data = {"raw": resp.text}

        # Mark backend unhealthy on hard failures so LB skips it next pick
        if resp.status_code >= 500:
            self.scheduler.set_healthy(worker_id, False)
            log.warning("Backend %s returned %s — marked unhealthy", worker_id, resp.status_code)

        self.scheduler.feedback(worker_id, e2e_ms, usage_tokens)

        if resp.status_code >= 400 and backend.api_style != "tgi_generate":
            return JSONResponse(status_code=resp.status_code, content=data)
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
                "decision": decision.as_dict(),
            }
        return JSONResponse(
            content=data,
            headers={
                "X-DIO-Backend": worker_id,
                "X-DIO-E2E-Ms": f"{e2e_ms:.1f}",
            },
        )

    async def _proxy_stream(
        self, body: Dict[str, Any], path: str, tier: str, allowed_backends: Optional[List[str]] = None
    ) -> StreamingResponse:
        prompt = _extract_prompt(body)
        tokens = self.token_counter.count(prompt, body)
        client = self._http()

        try:
            worker_id, decision = self.scheduler.pick(prompt, tier=tier, tokens=tokens, allowed_backends=allowed_backends)
        except AdmissionError as e:
            async def err_gen():
                yield f'data: {{"error": "{e}"}}\n\n'
            return StreamingResponse(
                err_gen(),
                status_code=503,
                media_type="text/event-stream",
                headers={"Retry-After": str(e.retry_after_sec)},
            )

        b = self.pool.get(worker_id)
        url = b.chat_url() if path == "chat" else b.completions_url()
        payload = dict(body)
        payload["stream"] = True
        if b.model:
            payload["model"] = b.model

        t0 = time.perf_counter()

        async def gen():
            try:
                async with client.stream("POST", url, json=payload, timeout=self.config.request_timeout_s) as resp:
                    if resp.status_code >= 500:
                        self.scheduler.set_healthy(worker_id, False)
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                e2e_ms = (time.perf_counter() - t0) * 1000.0
                self.scheduler.feedback(worker_id, e2e_ms, tokens)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "X-DIO-Backend": worker_id,
                "Cache-Control": "no-cache",
            },
        )

    async def _proxy_ollama_stream(
        self,
        body: Dict[str, Any],
        path: str,
        tier: str,
        is_chat: bool,
        allowed_backends: Optional[List[str]] = None,
    ) -> StreamingResponse:
        """
        Stream handler for Ollama native clients.

        Translates upstream OpenAI SSE chunks (data: {...}\n\n) into Ollama's
        newline-delimited JSON (ndjson) format on the fly.
        """
        model_name = body.get("model") or "dio-default"
        prompt = _extract_prompt(body)
        tokens = self.token_counter.count(prompt, body)
        client = self._http()

        try:
            worker_id, decision = self.scheduler.pick(
                prompt, tier=tier, tokens=tokens, allowed_backends=allowed_backends
            )
        except AdmissionError as e:
            async def err_gen():
                err_dict = {"error": str(e), "done": True}
                yield (json.dumps(err_dict) + "\n").encode("utf-8")

            return StreamingResponse(
                err_gen(),
                status_code=503,
                media_type="application/x-ndjson",
                headers={"Retry-After": str(e.retry_after_sec)},
            )

        b = self.pool.get(worker_id)
        url = b.chat_url() if path == "chat" else b.completions_url()
        payload = dict(body)
        payload["stream"] = True
        if b.model:
            payload["model"] = b.model

        t0 = time.perf_counter()

        async def gen():
            buffer = ""
            gen_tokens = 0
            try:
                async with client.stream(
                    "POST", url, json=payload, timeout=self.config.request_timeout_s
                ) as resp:
                    if resp.status_code >= 500:
                        self.scheduler.set_healthy(worker_id, False)

                    async for chunk_bytes in resp.aiter_bytes():
                        buffer += chunk_bytes.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices") or []
                                if not choices:
                                    continue
                                choice = choices[0]
                                if is_chat:
                                    delta = choice.get("delta") or {}
                                    delta_text = delta.get("content") or ""
                                    if delta_text:
                                        gen_tokens += 1
                                        ollama_chunk = {
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
                                        yield (json.dumps(ollama_chunk) + "\n").encode("utf-8")
                                else:
                                    delta_text = choice.get("text") or ""
                                    if delta_text:
                                        gen_tokens += 1
                                        ollama_chunk = {
                                            "model": model_name,
                                            "created_at": time.strftime(
                                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                            ),
                                            "response": delta_text,
                                            "done": False,
                                        }
                                        yield (json.dumps(ollama_chunk) + "\n").encode("utf-8")
                            except Exception:
                                continue

                e2e_ms = (time.perf_counter() - t0) * 1000.0
                total_duration_ns = int(e2e_ms * 1_000_000)
                final_chunk: Dict[str, Any] = {
                    "model": model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "done": True,
                    "total_duration": total_duration_ns,
                    "prompt_eval_count": tokens,
                    "eval_count": gen_tokens,
                }
                if is_chat:
                    final_chunk["message"] = {"role": "assistant", "content": ""}
                else:
                    final_chunk["response"] = ""
                yield (json.dumps(final_chunk) + "\n").encode("utf-8")
            finally:
                e2e_ms = (time.perf_counter() - t0) * 1000.0
                self.scheduler.feedback(worker_id, e2e_ms, tokens + gen_tokens)

        return StreamingResponse(
            gen(),
            media_type="application/x-ndjson",
            headers={
                "X-DIO-Backend": worker_id,
                "Cache-Control": "no-cache",
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
