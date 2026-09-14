"""Regressions for the hardening pass.

Each test here pins a defect found during the audit:
  * streaming must send ``Backend(api_key=...)`` like the JSON path already did;
  * a failed request must not be counted as a successful completion (goodput) nor
    train the NLMS learner;
  * one 5xx must not evict an otherwise healthy engine from the whole pool;
  * MCP notifications must not be answered and batches must be dispatched.
"""

from __future__ import annotations

import json

import httpx
import pytest

from dio.backends import Backend
from dio.gateway import DIOGateway


def _sse() -> bytes:
    frames = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
    ]
    return ("".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n").encode()


def _client(gw: DIOGateway) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://test")


# ---------------------------------------------------------------- backend auth
@pytest.mark.asyncio
async def test_streaming_paths_forward_backend_api_key():
    """Streaming used to omit the engine's bearer token, so token-gated engines
    401'd every stream while the JSON path worked."""
    seen: list = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        if request.headers.get("authorization") != "Bearer secret-token":
            return httpx.Response(401, json={"error": {"message": "missing bearer token"}})
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=_sse(), headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    gw = DIOGateway(
        backends=[
            Backend(id="b0", base_url="http://engine.invalid", tier="small", api_key="secret-token")
        ],
        admission_off=True,
    )
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async with _client(gw) as client:
        json_resp = await client.post(
            "/v1/chat/completions", json={"model": "default", "messages": [{"role": "user", "content": "hi"}]}
        )
        # Ollama native streams by default.
        ollama_resp = await client.post(
            "/api/chat", json={"model": "default", "messages": [{"role": "user", "content": "hi"}]}
        )
        sse_resp = await client.post(
            "/v1/chat/completions",
            json={"model": "default", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert json_resp.status_code == 200
    assert sse_resp.status_code == 200
    assert ollama_resp.status_code == 200
    assert seen and all(h == "Bearer secret-token" for h in seen), seen


# ------------------------------------------------------------------ accounting
@pytest.mark.asyncio
async def test_failed_requests_are_not_goodput_and_do_not_train_the_learner():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "engine exploded"}})

    gw = DIOGateway(
        backends=[Backend(id="b0", base_url="http://engine.invalid", tier="small")],
        admission_off=True,
    )
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async with _client(gw) as client:
        for _ in range(2):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 500

    adm = gw.scheduler.metrics()["admission"]
    assert adm["failed_total"] == 2
    assert adm["completed_total"] == 0
    assert adm["goodput_fraction"] == 0.0
    assert adm["failure_rate"] == 1.0
    # An error path is not a latency sample.
    assert gw.scheduler.metrics()["prediction"]["count"] == 0


# ---------------------------------------------------------------------- health
@pytest.mark.asyncio
async def test_single_5xx_does_not_evict_the_whole_pool():
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="overloaded")

    gw = DIOGateway(
        backends=[Backend(id="only", base_url="http://engine.invalid", tier="small")],
        admission_off=True,
    )
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async with _client(gw) as client:
        first = await client.post(
            "/v1/chat/completions", json={"model": "default", "messages": [{"role": "user", "content": "hi"}]}
        )
        # The second request must still reach the engine rather than being rejected
        # with "no healthy backends".
        second = await client.post(
            "/v1/chat/completions", json={"model": "default", "messages": [{"role": "user", "content": "hi"}]}
        )
        health_ok = (await client.get("/health")).json()

        # Third consecutive failure trips the breaker.
        third = await client.post(
            "/v1/chat/completions", json={"model": "default", "messages": [{"role": "user", "content": "hi"}]}
        )
        health_down = (await client.get("/health")).json()
        fourth = await client.post(
            "/v1/chat/completions", json={"model": "default", "messages": [{"role": "user", "content": "hi"}]}
        )

    assert first.status_code == 503
    assert second.status_code == 503
    assert calls["n"] == 3, calls  # the 4th never reached the engine
    assert health_ok["status"] == "ok"
    assert third.status_code == 503
    assert health_down["status"] == "degraded"
    assert health_down["backends_unhealthy"] == ["only"]
    assert fourth.status_code == 503
    assert fourth.json()["error"]["type"] == "dio_admission_rejected"


# ------------------------------------------------------------------- model map
def test_model_resolution_fails_closed_instead_of_substring_matching():
    gw = DIOGateway(
        backends=[],
        model_map={
            "meta-llama/Llama-3.2-3B-Instruct": ["gpu0"],
            "qwen2.5-7b": ["gpu1"],
        },
    )
    # Exact id and last-path-segment matches still work.
    assert gw._resolve_model_backends("meta-llama/Llama-3.2-3B-Instruct") == ["gpu0"]
    assert gw._resolve_model_backends("llama-3.2-3b-instruct") == ["gpu0"]
    # Prefix match on the segment name is preserved.
    assert gw._resolve_model_backends("qwen") == ["gpu1"]
    # Ambiguous short strings must not silently bind to an arbitrary backend.
    assert gw._resolve_model_backends("3B") == []
    assert gw._resolve_model_backends("a") == []
    assert gw._resolve_model_backends("totally-unknown") == []


# ------------------------------------------------------------------------ MCP
@pytest.mark.asyncio
async def test_mcp_notifications_get_no_reply_and_batches_are_dispatched():
    from dio.mcp import DIOMCPServer

    server = DIOMCPServer()
    try:
        # Notification (no "id") must produce no frame at all.
        assert await server.handle_message({"jsonrpc": "2.0", "method": "tools/list"}) is None
        assert await server.handle_message({"jsonrpc": "2.0", "method": "unknown/method"}) is None

        # Batch: one notification + one request -> exactly one reply.
        batch = await server.handle_message(
            [
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 7, "method": "ping"},
            ]
        )
        assert isinstance(batch, list)
        assert [r["id"] for r in batch] == [7]

        # A batch of only notifications produces nothing.
        assert await server.handle_message([{"jsonrpc": "2.0", "method": "ping"}]) is None

        # Non-object bodies must be Invalid Request, not a crash.
        for bad in (5, "hello"):
            resp = await server.handle_request(bad)
            assert resp["error"]["code"] == -32600

        # protocolVersion is echoed when supported.
        init = await server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            }
        )
        assert init["result"]["protocolVersion"] == "2025-06-18"

        # ...and falls back to a supported revision when it is not.
        init2 = await server.handle_request(
            {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}}
        )
        assert init2["result"]["protocolVersion"] in DIOMCPServer.SUPPORTED_PROTOCOL_VERSIONS
    finally:
        await server.close()
