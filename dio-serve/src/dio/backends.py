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
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import httpx

log = logging.getLogger("dio.backends")

ApiStyle = Literal[
    "openai",           # /v1/chat/completions + /v1/completions (default)
    "openai_chat",      # chat only
    "openai_completions",
    "tgi_generate",     # HuggingFace TGI /generate
    "custom",           # use chat_path / completions_path
]


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
    timeout_s: Optional[float] = None  # override global timeout

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + (path if path.startswith("/") else "/" + path)

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
    max_new = int(body.get("max_tokens") or body.get("max_new_tokens") or 64)
    return {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new,
            "temperature": float(body.get("temperature") or 0.7),
            "do_sample": True,
        },
    }


def tgi_generate_to_openai_chat(raw: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Normalize TGI response into OpenAI chat.completion shape for clients."""
    text = ""
    if isinstance(raw, list) and raw:
        text = raw[0].get("generated_text") or ""
    elif isinstance(raw, dict):
        text = raw.get("generated_text") or raw.get("text") or ""
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
            "prompt_tokens": 0,
            "completion_tokens": max(1, len(text) // 4),
            "total_tokens": max(1, len(text) // 4),
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
            "Registered backend %s → %s (tier=%s style=%s)",
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
        from typing import Any, Dict

        from fastapi import Body, FastAPI
        from fastapi.responses import JSONResponse

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
            sleep_ms = (80 + self.decode_ms * max_tokens) * self.latency_mult
            await asyncio.sleep(sleep_ms / 1000.0)
            text = f"[{self.name}] echo: {str(content)[:80]}"
            return JSONResponse(
                {
                    "id": f"chatcmpl-mock-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model") or "mock-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": tokens_in,
                        "completion_tokens": max_tokens,
                        "total_tokens": tokens_in + max_tokens,
                    },
                }
            )

        @app.post("/v1/completions")
        async def completions(body: Dict[str, Any] = Body(...)):
            prompt = body.get("prompt") or ""
            max_tokens = int(body.get("max_tokens") or 64)
            sleep_ms = (80 + self.decode_ms * max_tokens) * self.latency_mult
            await asyncio.sleep(sleep_ms / 1000.0)
            return JSONResponse(
                {
                    "id": f"cmpl-mock-{int(time.time()*1000)}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": body.get("model") or "mock-model",
                    "choices": [
                        {
                            "text": f"[{self.name}] {str(prompt)[:40]}",
                            "index": 0,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": max(1, len(str(prompt)) // 4),
                        "completion_tokens": max_tokens,
                        "total_tokens": max(1, len(str(prompt)) // 4) + max_tokens,
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
