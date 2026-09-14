"""Tests for Ollama native adapter (/api/chat, /api/generate, /api/tags, /api/version)."""

import json

import httpx
import pytest

from dio.backends import Backend, MockBackendServer
from dio.gateway import DIOGateway, _ollama_options_to_openai


def test_ollama_options_translation():
    opts = {
        "num_predict": 128,
        "temperature": 0.8,
        "top_p": 0.95,
        "stop": ["\n\n"],
        "presence_penalty": 0.2,
    }
    openai_params = _ollama_options_to_openai(opts)
    assert openai_params["max_tokens"] == 128
    assert openai_params["temperature"] == 0.8
    assert openai_params["top_p"] == 0.95
    assert openai_params["stop"] == ["\n\n"]
    assert openai_params["presence_penalty"] == 0.2


@pytest.mark.asyncio
async def test_ollama_metadata_endpoints():
    b0 = Backend(id="b0", base_url="http://127.0.0.1:19901", tier="small", model="llama3:latest")
    gw = DIOGateway(backends=[b0], model_map={"mistral": ["b0"]})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://test") as client:
        # /api/version
        v_resp = await client.get("/api/version")
        assert v_resp.status_code == 200
        assert "version" in v_resp.json()

        # /api/tags
        tags_resp = await client.get("/api/tags")
        assert tags_resp.status_code == 200
        models = tags_resp.json().get("models", [])
        names = {m["name"] for m in models}
        assert "mistral" in names
        assert "llama3:latest" in names

        # /api/show
        show_resp = await client.post("/api/show", json={"model": "llama3"})
        assert show_resp.status_code == 200
        assert "modelfile" in show_resp.json()


@pytest.mark.asyncio
async def test_ollama_chat_and_generate():
    server = MockBackendServer(port=19873, latency_mult=0.5, decode_ms_per_token=2.0, name="ollama_mock")
    await server.start()
    try:
        gw = DIOGateway(
            backends=[Backend(id="m0", base_url=server.base_url, tier="small")],
            admission_off=True,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://test") as client:
            # 1. Non-streaming chat
            chat_resp = await client.post(
                "/api/chat",
                json={
                    "model": "mock-llama",
                    "messages": [{"role": "user", "content": "say hello"}],
                    "stream": False,
                },
            )
            assert chat_resp.status_code == 200
            chat_data = chat_resp.json()
            assert chat_data["done"] is True
            assert chat_data["model"] == "mock-llama"
            assert "content" in chat_data["message"]
            assert chat_data["message"]["role"] == "assistant"
            assert "total_duration" in chat_data

            # 2. Streaming chat (ndjson)
            chat_stream_resp = await client.post(
                "/api/chat",
                json={
                    "model": "mock-llama",
                    "messages": [{"role": "user", "content": "stream to me"}],
                    "stream": True,
                },
            )
            assert chat_stream_resp.status_code == 200
            assert "application/x-ndjson" in chat_stream_resp.headers.get("content-type", "")

            lines = [line.strip() for line in chat_stream_resp.text.splitlines() if line.strip()]
            assert len(lines) >= 2
            parsed_lines = [json.loads(line) for line in lines]
            # Intermediate lines done=False
            assert any(p.get("done") is False for p in parsed_lines)
            # Final line done=True
            assert parsed_lines[-1]["done"] is True
            assert "total_duration" in parsed_lines[-1]

            # 3. Non-streaming generate
            gen_resp = await client.post(
                "/api/generate",
                json={
                    "model": "mock-llama",
                    "prompt": "generate something",
                    "stream": False,
                },
            )
            assert gen_resp.status_code == 200
            gen_data = gen_resp.json()
            assert gen_data["done"] is True
            assert "response" in gen_data
            assert "total_duration" in gen_data

            # 4. Streaming generate
            gen_stream_resp = await client.post(
                "/api/generate",
                json={
                    "model": "mock-llama",
                    "prompt": "stream generate",
                    "stream": True,
                },
            )
            assert gen_stream_resp.status_code == 200
            gen_lines = [line.strip() for line in gen_stream_resp.text.splitlines() if line.strip()]
            assert len(gen_lines) >= 2
            parsed_gen = [json.loads(line) for line in gen_lines]
            assert parsed_gen[-1]["done"] is True
    finally:
        await server.stop()


# --------------------------------------------------- request-level mapping
def test_ollama_request_mapping_forwards_format_and_tools():
    from dio.gateway import _ollama_request_to_openai

    # Ollama's "give me JSON" used to be dropped silently.
    assert _ollama_request_to_openai({"format": "json"}) == {
        "response_format": {"type": "json_object"}
    }

    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    mapped = _ollama_request_to_openai({"format": schema})
    assert mapped["response_format"]["type"] == "json_schema"
    assert mapped["response_format"]["json_schema"]["schema"] == schema

    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    mapped = _ollama_request_to_openai({"tools": tools, "tool_choice": "auto"})
    assert mapped["tools"] == tools
    assert mapped["tool_choice"] == "auto"

    # keep_alive is engine-local: it must not leak upstream as an Ollama-ism.
    assert _ollama_request_to_openai({"keep_alive": "5m"}) == {}
    assert _ollama_request_to_openai({}) == {}


def test_ollama_done_reason_maps_finish_reason():
    from dio.gateway import _ollama_done_reason

    assert _ollama_done_reason([{"finish_reason": "length"}]) == "length"
    assert _ollama_done_reason([{"finish_reason": "stop"}]) == "stop"
    assert _ollama_done_reason([{"finish_reason": "tool_calls"}]) == "tool_calls"
    assert _ollama_done_reason([]) == "stop"


@pytest.mark.asyncio
async def test_ollama_nonstream_reports_done_reason():
    srv = MockBackendServer(port=19912, latency_mult=1.0, decode_ms_per_token=1.0, name="gpu-y")
    await srv.start()
    try:
        gw = DIOGateway(backends=[Backend(id="gpu-y", base_url=srv.base_url)], admission_off=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gw.app), base_url="http://test"
        ) as client:
            chat = await client.post(
                "/api/chat",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )
            assert chat.status_code == 200
            body = chat.json()
            # Real Ollama always tells the client why generation stopped.
            assert body["done_reason"] in ("stop", "length")
            assert body["done"] is True

            gen = await client.post(
                "/api/generate", json={"model": "m", "prompt": "hi", "stream": False}
            )
            assert gen.status_code == 200
            assert gen.json()["done_reason"] in ("stop", "length")
    finally:
        await srv.stop()


@pytest.mark.asyncio
async def test_ollama_stream_final_chunk_has_completion_metadata():
    srv = MockBackendServer(port=19913, latency_mult=1.0, decode_ms_per_token=1.0, name="gpu-z")
    await srv.start()
    try:
        gw = DIOGateway(backends=[Backend(id="gpu-z", base_url=srv.base_url)], admission_off=True)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gw.app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST",
                "/api/chat",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            ) as r:
                assert r.status_code == 200
                lines = [
                    json.loads(line)
                    async for line in r.aiter_lines()
                    if line.strip()
                ]
        assert lines, "expected ndjson frames"
        assert not any(isinstance(ln, str) for ln in lines)
        final = lines[-1]
        assert final["done"] is True
        assert final["done_reason"] == "stop"
        assert final["load_duration"] >= 0
        assert "total_duration" in final and "eval_count" in final
    finally:
        await srv.stop()
