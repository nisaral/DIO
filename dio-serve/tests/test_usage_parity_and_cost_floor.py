"""Streamed token accounting + joint-cost floor (dogfooding regressions).

Both bugs came out of running a real agent swarm against one gateway:

* ``/api/chat`` and ``/api/generate`` reported *estimated* token counts on the
  streaming path (prompt chars/4, one per delta chunk) while the non-streaming
  path copied the engine's ``usage`` block, so the same request produced two
  different numbers depending only on ``stream``.
* The affinity/prefix bonus was subtracted unclamped, so a bonus (200 ms
  default) larger than ``exec_ms`` produced a negative joint cost. A negative
  score wins every comparison by an unbounded margin, which pins traffic to one
  worker for a reason that does not exist.
"""

import json
import os
import subprocess
import sys

import httpx
import pytest

from dio.backends import Backend, MockBackendServer
from dio.gateway import DIOGateway, _extract_prompt
from dio.scheduler import Scheduler

PORT_USAGE = 19941
PORT_LEGACY = 19942
PORT_PICKY = 19943


class _ScriptedEngine:
    """OpenAI-shaped engine used to pin the streaming edge cases.

    ``send_usage`` mirrors vLLM (honour ``stream_options.include_usage`` with a
    trailing usage chunk); ``reject_stream_options`` mirrors builds that reject
    the unknown field outright.
    """

    def __init__(
        self,
        port: int,
        *,
        send_usage: bool = False,
        reject_stream_options: bool = False,
    ) -> None:
        self.port = port
        self.send_usage = send_usage
        self.reject_stream_options = reject_stream_options
        self.requests = []
        self._server = None
        self._task = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _app(self):
        from fastapi import Body, FastAPI
        from fastapi.responses import JSONResponse, StreamingResponse

        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.post("/v1/chat/completions")
        async def chat(body: dict = Body(...)):
            self.requests.append(body)
            if self.reject_stream_options and "stream_options" in body:
                return JSONResponse(
                    {"error": {"message": "unexpected field stream_options"}}, status_code=400
                )
            if not body.get("stream"):
                return JSONResponse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "one two "},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                    }
                )

            async def gen():
                for word in ("one ", "two "):
                    yield (
                        'data: {"choices":[{"index":0,"delta":{"content":"%s"},"finish_reason":null}]}\n\n'
                        % word
                    )
                yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                if self.send_usage:
                    yield (
                        'data: {"choices":[],"usage":{"prompt_tokens":1,'
                        '"completion_tokens":2,"total_tokens":3}}\n\n'
                    )
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        return app

    async def start(self) -> None:
        import asyncio

        import uvicorn

        config = uvicorn.Config(self._app(), host="127.0.0.1", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(50):
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"{self.base_url}/health", timeout=0.5)
                    if r.status_code < 500:
                        return
            except Exception:
                pass
            await asyncio.sleep(0.1)
        raise RuntimeError("scripted engine did not start")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await self._task
            except Exception:
                pass


async def _ollama_stream(client: httpx.AsyncClient, payload: dict):
    """POST /api/chat and collect the ndjson frames into a list."""
    frames = []
    async with client.stream("POST", "/api/chat", json=payload) as resp:
        status = resp.status_code
        async for line in resp.aiter_lines():
            if line.strip():
                frames.append(json.loads(line))
    return status, frames


@pytest.mark.asyncio
async def test_ollama_streaming_reports_engine_usage_like_json():
    """stream=True and stream=False must agree on prompt_eval_count."""
    srv = MockBackendServer(
        port=PORT_USAGE, latency_mult=0.5, decode_ms_per_token=1.0, name="gpu-usage"
    )
    await srv.start()
    try:
        gw = DIOGateway(
            backends=[Backend(id="b0", base_url=srv.base_url, tier="small", model="mock-model")]
        )
        messages = [
            {"role": "system", "content": "s" * 400},
            {"role": "user", "content": "hello"},
        ]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gw.app), base_url="http://test"
        ) as client:
            status, frames = await _ollama_stream(
                client, {"model": "mock-model", "messages": messages, "stream": True}
            )
            assert status == 200
            non_streaming = (
                await client.post(
                    "/api/chat",
                    json={"model": "mock-model", "messages": messages, "stream": False},
                )
            ).json()
    finally:
        await srv.stop()

    final = frames[-1]
    assert final["done"] is True
    # The mock counts only the last message ("hello" -> 1). DIO's own character
    # estimate for the 400-char transcript would be two orders larger, so this
    # is the engine's number, not the fallback.
    assert non_streaming["prompt_eval_count"] == 1
    assert final["prompt_eval_count"] == 1
    assert final["eval_count"] >= 1


@pytest.mark.asyncio
async def test_ollama_stream_falls_back_to_estimates_without_engine_usage():
    """An engine that never sends usage must still produce a usable final frame."""
    srv = _ScriptedEngine(PORT_LEGACY, send_usage=False)
    await srv.start()
    try:
        gw = DIOGateway(backends=[Backend(id="b0", base_url=srv.base_url, tier="small", model="m")])
        body = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": True}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gw.app), base_url="http://test"
        ) as client:
            status, frames = await _ollama_stream(client, body)
    finally:
        await srv.stop()

    assert status == 200
    final = frames[-1]
    assert final["done"] is True
    # Two delta chunks -> chunk-count estimate; prompt -> router heuristic.
    assert final["eval_count"] == 2
    assert final["prompt_eval_count"] == gw.token_counter.prompt_tokens(_extract_prompt(body))


@pytest.mark.asyncio
async def test_ollama_stream_retries_without_stream_options_on_400():
    """A picky engine must cost one retry, not a broken stream."""
    srv = _ScriptedEngine(PORT_PICKY, reject_stream_options=True)
    await srv.start()
    try:
        gw = DIOGateway(backends=[Backend(id="b0", base_url=srv.base_url, tier="small", model="m")])
        body = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "stream": True}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gw.app), base_url="http://test"
        ) as client:
            status, frames = await _ollama_stream(client, body)
    finally:
        await srv.stop()

    assert status == 200
    assert frames[-1]["done"] is True
    assert len(srv.requests) == 2
    assert "stream_options" in srv.requests[0]
    assert "stream_options" not in srv.requests[1]


def test_joint_cost_is_never_negative_and_reports_the_clamp():
    s = Scheduler(strategy="nlms", admission_off=True, slo_ms=1e9, cache_bonus_ms=200.0)
    s.register("w0")
    pred = s.predictors["w0"]
    pred.fast_slope = 0.0
    pred.slow_slope = 0.0
    pred.intercept = 5.0

    _wid, first = s.pick("hello world", tokens=2)
    _wid, second = s.pick("hello world", tokens=2)

    assert first.affinity_hit is False
    assert second.affinity_hit is True
    assert second.total_ms >= 0.0
    # The 200 ms bonus may cancel at most the predicted service time.
    assert second.cache_bonus_ms <= second.exec_ms + second.wait_ms + 1e-9
    assert second.cache_bonus_ms + second.bonus_clamped_ms == pytest.approx(200.0)
    assert second.bonus_clamped_ms > 0.0
    assert s.metrics()["decisions"][-1]["bonus_clamped_ms"] > 0.0


def test_prefix_hash_is_stable_across_processes():
    """Affinity keys must not depend on PYTHONHASHSEED (builtin hash() does)."""
    text = "x" * 300
    local = Scheduler._prefix_hash(None, text)
    env = dict(os.environ, PYTHONHASHSEED="12345")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dio.scheduler import Scheduler; print(Scheduler._prefix_hash(None, 'x' * 300))",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout.strip()) == local
    # Documented truncation: the key covers the first 256 characters.
    assert Scheduler._prefix_hash(None, "a" * 256 + "1") == Scheduler._prefix_hash(
        None, "a" * 256 + "2"
    )
