"""
Backend registry — production load-balancing targets.

DIO forwards to **real** inference engines over HTTP. Mocks exist only for CI
and demos; production always points at live vLLM / SGLang / TGI / Ollama / etc.

Why the public surface looks like “OpenAI”
-----------------------------------------
Self-hosted engines almost universally expose the **OpenAI-compatible HTTP API**:

  • vLLM       → /v1/chat/completions, /v1/completions
  • SGLang     → OpenAI-compatible server
  • TGI        → OpenAI-compatible mode + /generate
  • Ollama     → /v1/chat/completions
  • LocalAI, LiteLLM, TensorRT-LLM OpenAI proxy, LM Studio, …

So one wire format covers **Llama, Mistral, Qwen, Phi, Gemma, …** — the *model*
is chosen by the engine’s ``model`` field, not by DIO.

DIO is **model-agnostic**: any weights the backend serves work. We also support
optional non-OpenAI paths (e.g. TGI ``/generate``) via ``Backend.api_style``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit

import httpx

log = logging.getLogger("dio.backends")

ApiStyle = Literal[
    "openai",           # /v1/chat/completions + /v1/completions (default)
    "openai_chat",      # chat only
    "openai_completions",
    "tgi_generate",     # HuggingFace TGI /generate
    "custom",           # use chat_path / completions_path
]


# Hosts that must never be reached by the gateway's own HTTP client. The admin
# plane can hot-register a backend, so an unvalidated ``base_url`` is an SSRF
# primitive: point DIO at cloud metadata and it will fetch that URL with its own
# network position. Only plain http(s) to a named host is ever forwarded.
_BLOCKED_BACKEND_HOSTS = frozenset(
    {"169.254.169.254", "fd00:ec2::254", "metadata.google.internal"}
)


def validate_backend_url(base_url: str, backend_id: str = "?") -> str:
    """Return a vetted backend URL, or raise ValueError explaining the problem."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(f"backend {backend_id!r}: base_url must be a non-empty URL")
    url = base_url.strip()
    if any(ch.isspace() for ch in url):
        raise ValueError(f"backend {backend_id!r}: base_url contains whitespace: {url!r}")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(
            f"backend {backend_id!r}: base_url must start with http:// or https:// "
            f"(got {parts.scheme or 'no scheme'!r})"
        )
    if not parts.hostname:
        raise ValueError(f"backend {backend_id!r}: base_url has no host: {url!r}")
    if parts.hostname in _BLOCKED_BACKEND_HOSTS:
        raise ValueError(
            f"backend {backend_id!r}: refusing to forward to metadata host {parts.hostname!r}"
        )
    return url


@dataclass
class Backend:
    """
    One **real** inference server (typically one GPU process).

    Production example::

        Backend(
            id="a100-0",
            base_url="http://10.0.1.5:8000",
            tier="large",
            model="meta-llama/Llama-3.1-8B-Instruct",  # optional override
            api_style="openai",
            api_key=None,  # or bearer token if engine requires it
        )
    """

    id: str
    base_url: str
    tier: str = "small"
    model: Optional[str] = None
    total_vram_mb: float = 24000.0
    free_vram_mb: float = 24000.0
    weight: float = 1.0
    prior_slope: Optional[float] = None
    prior_intercept: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    # Production knobs
    api_style: ApiStyle = "openai"
    api_key: Optional[str] = None
    chat_path: str = "/v1/chat/completions"
    completions_path: str = "/v1/completions"
    generate_path: str = "/generate"
    health_path: str = "/health"
    models_path: str = "/v1/models"
    metrics_path: str = "/metrics"  # vLLM Prometheus (non-invasive)
    timeout_s: Optional[float] = None  # override global timeout

    def __post_init__(self) -> None:
        self.base_url = validate_backend_url(self.base_url, self.id)

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + (path if path.startswith("/") else "/" + path)

    def metrics_url(self) -> str:
        return self._url(self.metrics_path)

    def chat_url(self) -> str:
        if self.api_style == "tgi_generate":
            return self._url(self.generate_path)
        if self.api_style == "custom":
            return self._url(self.chat_path)
        return self._url(self.chat_path)

    def completions_url(self) -> str:
        if self.api_style == "tgi_generate":
            return self._url(self.generate_path)
        if self.api_style == "custom":
            return self._url(self.completions_path)
        return self._url(self.completions_path)

    def models_url(self) -> str:
        return self._url(self.models_path)

    def health_url(self) -> str:
        return self._url(self.health_path)

    def auth_headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}


def openai_chat_to_tgi_generate(body: Dict[str, Any]) -> Dict[str, Any]:
    """Map OpenAI chat payload → TGI /generate body (best-effort)."""
    messages = body.get("messages") or []
    if messages:
        parts = []
        for m in messages:
            parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
        prompt = "\n".join(parts) + "\nassistant:"
    else:
        prompt = str(body.get("prompt") or "")
    # NOTE: "or" would treat an explicit 0 (greedy decoding, max_tokens=0 meaning
    # "engine default") as absent and silently substitute sampling defaults.
    max_new = body.get("max_tokens")
    if max_new is None:
        max_new = body.get("max_new_tokens")
    params: Dict[str, Any] = {
        "max_new_tokens": int(max_new) if max_new is not None else 64,
        "do_sample": True,
    }
    temperature = body.get("temperature")
    if temperature is not None:
        temperature = float(temperature)
        params["temperature"] = temperature
        # temperature == 0 means greedy; TGI only samples when do_sample is true.
        params["do_sample"] = temperature > 0.0
    else:
        params["temperature"] = 0.7
    for src, dst in (
        ("top_p", "top_p"),
        ("presence_penalty", "repetition_penalty"),
        ("frequency_penalty", "frequency_penalty"),
    ):
        if body.get(src) is not None:
            params[dst] = float(body[src])
    if body.get("stop") is not None:
        stop = body["stop"]
        params["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    return {"inputs": prompt, "parameters": params}


def tgi_generate_to_openai_chat(
    raw: Dict[str, Any],
    model: str,
    prompt_tokens: int = 0,
) -> Dict[str, Any]:
    """Normalize TGI response into OpenAI chat.completion shape for clients.

    TGI does not report an OpenAI-style ``usage`` block, so token counts are
    filled in here: ``prompt_tokens`` from the caller (which knows the prompt),
    ``completion_tokens`` from TGI's own ``details.generated_tokens`` when the
    server reports it, else a len/4 heuristic. Reporting ``prompt_tokens: 0``
    made cost dashboards and SDK token accounting wrong.
    """
    text = ""
    details: Dict[str, Any] = {}
    if isinstance(raw, list) and raw:
        text = raw[0].get("generated_text") or ""
        if isinstance(raw[0].get("details"), dict):
            details = raw[0]["details"]
    elif isinstance(raw, dict):
        text = raw.get("generated_text") or raw.get("text") or ""
        if isinstance(raw.get("details"), dict):
            details = raw["details"]
    completion_tokens = details.get("generated_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        completion_tokens = max(1, len(text) // 4)
    prompt_n = int(prompt_tokens) if isinstance(prompt_tokens, int) and prompt_tokens > 0 else 0
    return {
        "id": f"chatcmpl-tgi-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_n,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_n + completion_tokens,
        },
    }


class BackendPool:
    """Registry of production backends + HTTP forward helpers."""

    def __init__(self, backends: Optional[List[Backend]] = None) -> None:
        self.backends: Dict[str, Backend] = {}
        for b in backends or []:
            self.add(b)

    def add(self, backend: Backend) -> None:
        self.backends[backend.id] = backend
        log.info(
            "Registered backend %s -> %s (tier=%s style=%s)",
            backend.id,
            backend.base_url,
            backend.tier,
            backend.api_style,
        )

    def get(self, backend_id: str) -> Backend:
        return self.backends[backend_id]

    def list(self) -> List[Backend]:
        return list(self.backends.values())

    async def probe_health(self, client: httpx.AsyncClient, backend_id: str) -> bool:
        b = self.backends[backend_id]
        headers = b.auth_headers()
        for url in (b.health_url(), b.models_url()):
            try:
                r = await client.get(url, headers=headers, timeout=3.0)
                if r.status_code < 500:
                    return True
            except Exception:
                continue
        return False

    async def forward_chat(
        self,
        client: httpx.AsyncClient,
        backend_id: str,
        body: Dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        b = self.backends[backend_id]
        headers = b.auth_headers()
        t = b.timeout_s if b.timeout_s is not None else timeout

        if b.api_style == "tgi_generate":
            payload = openai_chat_to_tgi_generate(body)
            return await client.post(b.chat_url(), json=payload, headers=headers, timeout=t)

        payload = dict(body)
        if b.model:
            payload["model"] = b.model
        return await client.post(b.chat_url(), json=payload, headers=headers, timeout=t)

    async def forward_completions(
        self,
        client: httpx.AsyncClient,
        backend_id: str,
        body: Dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        b = self.backends[backend_id]
        headers = b.auth_headers()
        t = b.timeout_s if b.timeout_s is not None else timeout

        if b.api_style == "tgi_generate":
            payload = openai_chat_to_tgi_generate(body)
            return await client.post(b.completions_url(), json=payload, headers=headers, timeout=t)

        payload = dict(body)
        if b.model:
            payload["model"] = b.model
        return await client.post(b.completions_url(), json=payload, headers=headers, timeout=t)


class MockBackendServer:
    """
    In-process fake OpenAI server for **CI / demos only**.

    Production: do not use this — register real ``Backend(base_url=...)`` pointing
    at vLLM (or other engines).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9001,
        latency_mult: float = 1.0,
        decode_ms_per_token: float = 12.0,
        name: str = "mock",
    ) -> None:
        self.host = host
        self.port = port
        self.latency_mult = latency_mult
        self.decode_ms = decode_ms_per_token
        self.name = name
        self._server = None
        self._task = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _app(self):

        from fastapi import Body, FastAPI
        from fastapi.responses import JSONResponse, StreamingResponse

        app = FastAPI(title=f"DIO Mock Backend ({self.name})")

        @app.get("/health")
        async def health():
            return {"status": "ok", "backend": self.name}

        @app.get("/v1/models")
        async def models():
            return {
                "object": "list",
                "data": [{"id": "mock-model", "object": "model", "owned_by": "dio"}],
            }

        @app.post("/v1/chat/completions")
        async def chat(body: Dict[str, Any] = Body(...)):
            messages = body.get("messages") or []
            content = messages[-1]["content"] if messages else ""
            max_tokens = int(body.get("max_tokens") or 64)
            tokens_in = max(1, len(str(content)) // 4)
            model_name = body.get("model") or "mock-model"

            if body.get("stream"):
                async def chat_stream():
                    cid = f"chatcmpl-mock-{int(time.time()*1000)}"
                    created = int(time.time())
                    # Initial TTFT delay
                    await asyncio.sleep(max(0.01, (40 * self.latency_mult) / 1000.0))
                    # Role chunk
                    init_chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(init_chunk)}\n\n"

                    text = f"[{self.name}] echo: {str(content)[:80]}"
                    words = text.split(" ")[:max_tokens]
                    # A real engine reports the tokens it produced and why it
                    # stopped; reporting max_tokens as completion_tokens made the
                    # streamed and non-streamed paths disagree by design.
                    truncated = len(text.split(" ")) > max_tokens
                    finish = "length" if truncated else "stop"
                    for i, word in enumerate(words):
                        chunk_text = word + (" " if i < len(words) - 1 else "")
                        chunk = {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(max(0.005, (self.decode_ms * self.latency_mult) / 1000.0))

                    stop_chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": finish}
                        ],
                    }
                    yield f"data: {json.dumps(stop_chunk)}\n\n"
                    # OpenAI stream_options.include_usage: a trailing chunk with
                    # empty choices carries the engine's accounting, so a
                    # translating gateway can report real tokens for streams too.
                    if (body.get("stream_options") or {}).get("include_usage"):
                        usage_chunk = {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": tokens_in,
                                "completion_tokens": len(words),
                                "total_tokens": tokens_in + len(words),
                            },
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(chat_stream(), media_type="text/event-stream")

            # Cap the *simulated* decode work (see /v1/completions): a legal but
            # absurd max_tokens must not park the mock for days -- real engines
            # stop at max_model_len and report finish_reason "length".
            sleep_ms = (80 + self.decode_ms * min(max_tokens, 4096)) * self.latency_mult
            await asyncio.sleep(sleep_ms / 1000.0)
            text = f"[{self.name}] echo: {str(content)[:80]}"
            words = text.split(" ")
            truncated = len(words) > max_tokens
            if truncated:
                text = " ".join(words[:max_tokens])
            generated = min(max_tokens, len(words))
            return JSONResponse(
                {
                    "id": f"chatcmpl-mock-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "length" if truncated else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": tokens_in,
                        "completion_tokens": generated,
                        "total_tokens": tokens_in + generated,
                    },
                }
            )

        @app.post("/v1/completions")
        async def completions(body: Dict[str, Any] = Body(...)):
            prompt = body.get("prompt") or ""
            max_tokens = int(body.get("max_tokens") or 64)
            model_name = body.get("model") or "mock-model"

            if body.get("stream"):
                async def comp_stream():
                    cid = f"cmpl-mock-{int(time.time()*1000)}"
                    created = int(time.time())
                    await asyncio.sleep(max(0.01, (40 * self.latency_mult) / 1000.0))
                    text = f"[{self.name}] {str(prompt)[:40]}"
                    words = text.split(" ")[:max_tokens]
                    truncated = len(text.split(" ")) > max_tokens
                    finish = "length" if truncated else "stop"
                    for i, word in enumerate(words):
                        chunk_text = word + (" " if i < len(words) - 1 else "")
                        chunk = {
                            "id": cid,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {"text": chunk_text, "index": 0, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(max(0.005, (self.decode_ms * self.latency_mult) / 1000.0))

                    stop_chunk = {
                        "id": cid,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"text": "", "index": 0, "finish_reason": finish}
                        ],
                    }
                    yield f"data: {json.dumps(stop_chunk)}\n\n"
                    if (body.get("stream_options") or {}).get("include_usage"):
                        tokens_in = max(1, len(str(prompt)) // 4)
                        usage_chunk = {
                            "id": cid,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": tokens_in,
                                "completion_tokens": len(words),
                                "total_tokens": tokens_in + len(words),
                            },
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(comp_stream(), media_type="text/event-stream")

            sleep_ms = (80 + self.decode_ms * min(max_tokens, 4096)) * self.latency_mult
            await asyncio.sleep(sleep_ms / 1000.0)
            tokens_in = max(1, len(str(prompt)) // 4)
            text = f"[{self.name}] {str(prompt)[:40]}"
            words = text.split(" ")
            truncated = len(words) > max_tokens
            if truncated:
                text = " ".join(words[:max_tokens])
            generated = min(max_tokens, len(words))
            return JSONResponse(
                {
                    "id": f"cmpl-mock-{int(time.time()*1000)}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "text": text,
                            "index": 0,
                            "finish_reason": "length" if truncated else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": tokens_in,
                        "completion_tokens": generated,
                        "total_tokens": tokens_in + generated,
                    },
                }
            )

        return app

    async def start(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            self._app(),
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(50):
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"{self.base_url}/health", timeout=0.5)
                    if r.status_code == 200:
                        log.info("Mock backend %s up at %s (CI/demo only)", self.name, self.base_url)
                        return
            except Exception:
                await asyncio.sleep(0.1)
        log.warning("Mock backend %s may not be ready", self.name)

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except Exception:
                pass
