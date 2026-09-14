"""Tests for SSE streaming passthrough and mock streaming."""

import json
import pytest
import httpx
from dio.backends import Backend, MockBackendServer
from dio.gateway import DIOGateway


@pytest.mark.asyncio
async def test_mock_backend_streaming():
    server = MockBackendServer(port=19871, latency_mult=0.5, decode_ms_per_token=2.0, name="stream_mock")
    await server.start()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test chat streaming
            resp = await client.post(
                f"{server.base_url}/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello world"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            chunks = []
            saw_done = False
            for line in resp.text.split("\n"):
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                parsed = json.loads(data)
                chunks.append(parsed)

            assert saw_done is True
            assert len(chunks) >= 2
            # First chunk has role
            assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
            # Intermediate chunks have content
            has_content = any("content" in c["choices"][0]["delta"] for c in chunks[1:])
            assert has_content is True

            # Test text completions streaming
            c_resp = await client.post(
                f"{server.base_url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "completion test prompt",
                    "stream": True,
                },
            )
            assert c_resp.status_code == 200
            assert "text/event-stream" in c_resp.headers.get("content-type", "")
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_gateway_stream_passthrough():
    server = MockBackendServer(port=19872, latency_mult=0.5, decode_ms_per_token=2.0, name="gw_stream")
    await server.start()
    try:
        gw = DIOGateway(
            backends=[Backend(id="s0", base_url=server.base_url, tier="small")],
            admission_off=True,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "stream through gateway"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            assert resp.headers.get("X-DIO-Backend") == "s0"

            body_text = resp.text
            assert "data: [DONE]" in body_text

            collected_content = ""
            for line in body_text.splitlines():
                line = line.strip()
                if line.startswith("data:") and not line.endswith("[DONE]"):
                    data = json.loads(line[5:].strip())
                    for c in data.get("choices", []):
                        collected_content += c.get("delta", {}).get("content", "")
            assert "stream through gateway" in collected_content

            # Verify scheduler recorded feedback
            metrics = gw.scheduler.metrics()
            assert metrics["workers"]["s0"]["updates"] >= 1
    finally:
        await server.stop()
