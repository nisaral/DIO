"""Regressions for the streaming proxy.

Covers three defects found by auditing the hot path:
  1. the admission-rejection generator raised NameError because it closed over an
     ``except ... as e`` variable that Python unbinds when the except block exits;
  2. a non-2xx engine response was flattened into ``200 OK`` with an error body
     inside the stream;
  3. the NLMS learner was updated with the backend-reported token count instead of
     the token feature the predictor actually used.
"""

from __future__ import annotations

import json

import httpx
import pytest

from dio.backends import Backend
from dio.gateway import DIOGateway


def _gw(**overrides) -> DIOGateway:
    return DIOGateway(
        backends=[Backend(id="b0", base_url="http://engine.invalid", tier="small")],
        **overrides,
    )


def _client(gw: DIOGateway) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://test")


def _sse(text: str = "hello") -> bytes:
    frames = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames)
    return (body + "data: [DONE]\n\n").encode()


# --------------------------------------------------------------------------- 1
@pytest.mark.asyncio
async def test_streaming_admission_rejection_is_valid_sse():
    # Force a real admission rejection: a 1 ms absolute budget rejects the
    # cold-start prediction. (An *unknown model* is a 404 now, not a 503.)
    gw = _gw(slo_ms=1.0, admission_mode="absolute")
    async with _client(gw) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert resp.status_code == 503
    assert resp.headers.get("retry-after")
    payloads = [ln[5:].strip() for ln in resp.text.splitlines() if ln.startswith("data:")]
    assert payloads[-1] == "[DONE]"
    body = json.loads(payloads[0])
    assert body["error"]["type"] == "dio_admission_rejected"
    assert body["error"]["code"] == "service_unavailable"


@pytest.mark.asyncio
async def test_ollama_stream_admission_rejection_is_valid_ndjson():
    gw = _gw(slo_ms=1.0, admission_mode="absolute")
    async with _client(gw) as client:
        resp = await client.post(
            "/api/chat",
            json={"model": "default", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 503
    assert resp.headers.get("retry-after")
    line = json.loads(resp.text.strip().splitlines()[0])
    assert line["done"] is True
    assert line["error"]


# --------------------------------------------------------------------------- 2
@pytest.mark.asyncio
async def test_streaming_propagates_upstream_error_status():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    gw = _gw()
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with _client(gw) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "default", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert resp.status_code == 404
    assert resp.json()["error"]["message"] == "model not found"


@pytest.mark.asyncio
async def test_ollama_stream_propagates_upstream_error_status():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    gw = _gw()
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with _client(gw) as client:
        resp = await client.post(
            "/api/chat", json={"model": "default", "messages": [{"role": "user", "content": "hi"}]}
        )
    assert resp.status_code == 404
    line = json.loads(resp.text.strip().splitlines()[0])
    assert line["done"] is True
    assert "model not found" in line["error"]


@pytest.mark.asyncio
async def test_streaming_success_still_passes_through():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_sse("streamed text"), headers={"content-type": "text/event-stream"}
        )

    gw = _gw()
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with _client(gw) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "default", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert resp.status_code == 200
    assert resp.headers.get("X-DIO-Backend") == "b0"
    assert "data: [DONE]" in resp.text
    assert "streamed text" in resp.text


# --------------------------------------------------------------------------- 3
@pytest.mark.asyncio
async def test_learner_uses_prediction_feature_not_backend_usage():
    """The NLMS update must use the same token feature the predictor saw."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 9999, "total_tokens": 9999},
            },
        )

    gw = _gw(admission_off=True)
    gw._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with _client(gw) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hello there"}],
                "max_tokens": 8,
            },
        )
    assert resp.status_code == 200
    payload = resp.json()["dio"]
    # Backend usage is surfaced for observability...
    assert payload["reported_tokens"] == 9999
    # ...but the learner is fed the prediction-time feature, not the engine report.
    assert payload["tokens"] != 9999
    sample = gw.scheduler.metrics()["prediction"]["samples"][-1]
    assert sample["tokens"] == payload["tokens"]
