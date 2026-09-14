"""Tests for Ollama native adapter (/api/chat, /api/generate, /api/tags, /api/version)."""

import json
import pytest
import httpx
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

            lines = [l.strip() for l in chat_stream_resp.text.splitlines() if l.strip()]
            assert len(lines) >= 2
            parsed_lines = [json.loads(l) for l in lines]
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
            gen_lines = [l.strip() for l in gen_stream_resp.text.splitlines() if l.strip()]
            assert len(gen_lines) >= 2
            parsed_gen = [json.loads(l) for l in gen_lines]
            assert parsed_gen[-1]["done"] is True
    finally:
        await server.stop()
