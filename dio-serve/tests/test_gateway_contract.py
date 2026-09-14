"""Gateway contract: error envelope, model ids, budget headers, SSE->ndjson.

Every case here is a defect that a dogfooding agent hit while driving one live
gateway (see docs/launch): a malformed body reporting a framework 422, an
unknown model name served as a 200, a legal-looking ``max_tokens`` poisoning the
learned prediction, streamed tool calls vanishing, and ``done_reason`` always
saying ``stop``.
"""

import asyncio
import json

import httpx
import pytest

from dio.backends import Backend, MockBackendServer
from dio.gateway import DIOGateway

PORT_CAP = 19951
PORT_MODEL = 19952
PORT_TOOLS = 19953
PORT_PLAIN = 19954


class _ScriptedEngine:
    """Minimal OpenAI-shaped engine: streams, tool calls, picky 400s."""

    def __init__(
        self,
        port: int,
        *,
        send_usage: bool = False,
        reject_stream_options: bool = False,
        tool_calls: bool = False,
        finish_reason: str = "stop",
    ) -> None:
        self.port = port
        self.send_usage = send_usage
        self.reject_stream_options = reject_stream_options
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason
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
                                "finish_reason": self.finish_reason,
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                    }
                )

            async def gen():
                def frame(delta, finish=None):
                    return "data: " + json.dumps(
                        {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    ) + "\n\n"

                if self.tool_calls:
                    yield frame(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city": '},
                                }
                            ]
                        }
                    )
                    yield frame(
                        {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}
                    )
                else:
                    for word in ("one ", "two "):
                        yield frame({"content": word})
                yield frame({}, self.finish_reason)
                if self.send_usage:
                    yield "data: " + json.dumps(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 2,
                                "total_tokens": 3,
                            },
                        }
                    ) + "\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        return app

    async def start(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            self._app(), host="127.0.0.1", port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(50):
            try:
                async with httpx.AsyncClient() as c:
                    if (await c.get(f"{self.base_url}/health", timeout=0.5)).status_code < 500:
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


def _client(gw: DIOGateway) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://test")


async def _stream_frames(client: httpx.AsyncClient, payload: dict):
    frames = []
    async with client.stream("POST", "/api/chat", json=payload) as resp:
        status = resp.status_code
        headers = dict(resp.headers)
        async for line in resp.aiter_lines():
            if line.strip():
                frames.append(json.loads(line))
    return status, headers, frames


@pytest.mark.asyncio
async def test_malformed_json_is_400_not_422():
    gw = DIOGateway(backends=[Backend(id="b0", base_url="http://127.0.0.1:1")])
    async with _client(gw) as client:
        r = await client.post(
            "/v1/chat/completions",
            content=b'{"model": "m", ',
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_body"


@pytest.mark.asyncio
async def test_float_max_tokens_says_integer_not_positive():
    gw = DIOGateway(backends=[Backend(id="b0", base_url="http://127.0.0.1:1")])
    async with _client(gw) as client:
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 1e9},
        )
    assert r.status_code == 400
    # 1e9 parses as a float, so "must be a positive integer" would send the
    # caller after the wrong bug.
    assert r.json()["error"]["message"] == "'max_tokens' must be an integer (got 1000000000.0)"


@pytest.mark.asyncio
async def test_unknown_model_is_404_when_a_model_map_exists():
    srv = MockBackendServer(
        port=PORT_MODEL, latency_mult=0.01, decode_ms_per_token=0.01, name="gpu-known"
    )
    await srv.start()
    try:
        b = Backend(id="b0", base_url=srv.base_url, tier="small", model="known-model")
        gw = DIOGateway(backends=[b], model_map={"known-model": ["b0"]})
        async with _client(gw) as client:
            unknown = await client.post(
                "/v1/chat/completions",
                json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
            )
            known = await client.post(
                "/v1/chat/completions",
                json={"model": "known-model", "messages": [{"role": "user", "content": "hi"}]},
            )
            ollama_unknown = await client.post(
                "/api/chat", json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]}
            )
    finally:
        await srv.stop()

    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "model_not_found"
    assert known.status_code == 200
    assert ollama_unknown.status_code == 404
    assert "not found" in ollama_unknown.json()["error"]


@pytest.mark.asyncio
async def test_absurd_max_tokens_cannot_poison_the_prediction():
    srv = MockBackendServer(
        port=PORT_CAP, latency_mult=0.001, decode_ms_per_token=0.01, name="gpu-cap"
    )
    await srv.start()
    try:
        b = Backend(id="b0", base_url=srv.base_url, tier="small", model="mock-model")
        gw = DIOGateway(backends=[b], token_feature_cap=256)
        async with _client(gw) as client:
            r = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 10**9,
                },
            )
    finally:
        await srv.stop()

    assert r.status_code == 200
    decision = r.json()["dio"]["decision"]
    # The request is still forwarded with the client's max_tokens; only the
    # routing feature (and therefore the learned prediction) is bounded.
    assert decision["tokens"] == 256
    assert decision["exec_ms"] < 1e6
    m = gw.scheduler.metrics()
    # Pre-fix this one request left mae_ms in the tens of millions and mape in
    # the millions of percent (see docs/launch/evidence); the point of the clamp
    # is that a single sample stays a prediction error, not a poisoned statistic.
    assert m["prediction"]["mae_ms"] < 1e5
    assert m["prediction"]["mape_pct"] < 1e5


@pytest.mark.asyncio
async def test_streamed_tool_calls_reach_ollama_clients():
    srv = _ScriptedEngine(PORT_TOOLS, tool_calls=True, finish_reason="tool_calls")
    await srv.start()
    try:
        gw = DIOGateway(backends=[Backend(id="b0", base_url=srv.base_url, tier="small", model="m")])
        async with _client(gw) as client:
            status, _headers, frames = await _stream_frames(
                client,
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "stream": True,
                },
            )
    finally:
        await srv.stop()

    assert status == 200
    calls = [
        call
        for frame in frames
        for call in (frame.get("message") or {}).get("tool_calls") or []
    ]
    assert calls, "tool call fragments were dropped"
    assert calls[-1]["function"]["name"] == "get_weather"
    assert calls[-1]["function"]["arguments"] == '{"city": "Paris"}'
    assert frames[-1]["done_reason"] == "tool_calls"
    assert frames[-1]["message"]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'


@pytest.mark.asyncio
async def test_streamed_done_reason_follows_engine_finish_reason():
    srv = _ScriptedEngine(PORT_PLAIN, finish_reason="length")
    await srv.start()
    try:
        gw = DIOGateway(backends=[Backend(id="b0", base_url=srv.base_url, tier="small", model="m")])
        async with _client(gw) as client:
            _status, headers, frames = await _stream_frames(
                client, {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
            )
    finally:
        await srv.stop()

    assert frames[-1]["done_reason"] == "length"
    # Budget/backpressure signal is known before the first byte, so a stream can
    # carry it even though its own latency is not measured yet.
    assert float(headers["x-dio-predicted-ms"]) >= 0.0
    assert float(headers["x-dio-budget-ms"]) == gw.config.slo_ms


@pytest.mark.asyncio
async def test_json_responses_carry_budget_and_over_budget_headers():
    srv = MockBackendServer(
        port=PORT_PLAIN + 1, latency_mult=0.01, decode_ms_per_token=0.01, name="gpu-budget"
    )
    await srv.start()
    try:
        gw = DIOGateway(
            backends=[Backend(id="b0", base_url=srv.base_url, tier="small", model="mock-model")],
            slo_ms=1.0,
        )
        async with _client(gw) as client:
            r = await client.post(
                "/v1/chat/completions",
                json={"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]},
            )
    finally:
        await srv.stop()

    assert r.status_code == 200
    assert float(r.headers["X-DIO-Budget-Ms"]) == 1.0
    assert r.headers["X-DIO-Over-Budget"] == "1"
    assert r.json()["dio"]["over_budget"] is True
