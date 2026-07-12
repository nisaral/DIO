"""Backend registry — any OpenAI-compatible HTTP server (vLLM, SGLang, TGI, Ollama)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("dio.backends")


@dataclass
class Backend:
    """One inference engine instance (typically one vLLM process / GPU)."""

    id: str
    base_url: str
    tier: str = "small"
    model: Optional[str] = None  # if set, override request model
    total_vram_mb: float = 24000.0
    free_vram_mb: float = 24000.0
    weight: float = 1.0
    # Optional static latency prior for STATIC strategy warm-start
    prior_slope: Optional[float] = None
    prior_intercept: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)

    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    def completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/completions"

    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/models"

    def health_url(self) -> str:
        # vLLM / OpenAI servers usually expose /health or /v1/models
        return self.base_url.rstrip("/") + "/health"


class BackendPool:
    def __init__(self, backends: Optional[List[Backend]] = None) -> None:
        self.backends: Dict[str, Backend] = {}
        for b in backends or []:
            self.add(b)

    def add(self, backend: Backend) -> None:
        self.backends[backend.id] = backend
        log.info("Registered backend %s → %s (tier=%s)", backend.id, backend.base_url, backend.tier)

    def get(self, backend_id: str) -> Backend:
        return self.backends[backend_id]

    def list(self) -> List[Backend]:
        return list(self.backends.values())

    async def probe_health(self, client: httpx.AsyncClient, backend_id: str) -> bool:
        b = self.backends[backend_id]
        for url in (b.health_url(), b.models_url()):
            try:
                r = await client.get(url, timeout=3.0)
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
        payload = dict(body)
        if b.model:
            payload["model"] = b.model
        # Prefer chat; some engines only have completions — caller can fall back.
        return await client.post(b.chat_url(), json=payload, timeout=timeout)

    async def forward_completions(
        self,
        client: httpx.AsyncClient,
        backend_id: str,
        body: Dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        b = self.backends[backend_id]
        payload = dict(body)
        if b.model:
            payload["model"] = b.model
        return await client.post(b.completions_url(), json=payload, timeout=timeout)


class MockBackendServer:
    """
    In-process fake OpenAI server for zero-GPU demos and CI.

    Usage::

        mock = MockBackendServer(port=9001, latency_mult=1.0)
        await mock.start()
        # ... use http://127.0.0.1:9001 as backend
        await mock.stop()
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
        from fastapi import FastAPI, Request
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
        async def chat(request: Request):
            body = await request.json()
            messages = body.get("messages") or []
            content = messages[-1]["content"] if messages else ""
            max_tokens = int(body.get("max_tokens") or 64)
            tokens_in = max(1, len(content) // 4)
            # Simulate decode cost
            sleep_ms = (80 + self.decode_ms * max_tokens) * self.latency_mult
            await asyncio.sleep(sleep_ms / 1000.0)
            text = f"[{self.name}] echo: {content[:80]}"
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
        async def completions(request: Request):
            body = await request.json()
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
                    "choices": [{"text": f"[{self.name}] {prompt[:40]}", "index": 0, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": max(1, len(prompt) // 4),
                        "completion_tokens": max_tokens,
                        "total_tokens": max(1, len(prompt) // 4) + max_tokens,
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
        # wait until port listens
        for _ in range(50):
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"{self.base_url}/health", timeout=0.5)
                    if r.status_code == 200:
                        log.info("Mock backend %s up at %s", self.name, self.base_url)
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
