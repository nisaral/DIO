"""Regressions for the second hardening pass.

Pins three defects found by dogfooding the gateway:

  * ``Backend.base_url`` was unvalidated, and the admin plane can hot-register a
    backend -- so ``POST /debug/backends`` was an SSRF primitive against the
    cloud metadata service;
  * TGI translations always reported ``prompt_tokens: 0``, so SDK token
    accounting and cost dashboards were wrong for TGI fleets;
  * ``Retry-After: 1`` on an empty pool understated the real recovery time (the
    health probe interval), so clients retried straight into another 503;
  * the Ollama *non-streaming* translations dropped DIO's routing headers, so
    Ollama clients could not see which engine served the request.
"""

from __future__ import annotations

import httpx
import pytest

from dio.backends import Backend, tgi_generate_to_openai_chat, validate_backend_url
from dio.config_file import ConfigError, load_config_file
from dio.gateway import DIOGateway


def _client(gw: DIOGateway) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://test")


# ------------------------------------------------------------- base_url rules
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000",
        "https://gpu-1.internal:8080",
        "http://localhost:11434",
        "http://10.0.1.5:8000",
    ],
)
def test_valid_backend_urls_are_accepted(url):
    assert validate_backend_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "gpu-1:8000",
        "file:///etc/passwd",
        "ftp://gpu-1/x",
        "http://",
        "http://169.254.169.254/latest/meta-data/",
        "http://user name@host/",
    ],
)
def test_invalid_backend_urls_are_rejected(url):
    with pytest.raises(ValueError):
        validate_backend_url(url)


def test_backend_validates_url_on_construction():
    with pytest.raises(ValueError):
        Backend(id="evil", base_url="file:///etc/passwd")


@pytest.mark.asyncio
async def test_debug_registration_rejects_ssrf_and_bad_input():
    gw = DIOGateway(
        backends=[Backend(id="b0", base_url="http://127.0.0.1:19011")],
        admission_off=True,
    )
    async with _client(gw) as client:
        ssrf = await client.post(
            "/debug/backends",
            json={"id": "evil", "base_url": "http://169.254.169.254/latest/meta-data"},
        )
        assert ssrf.status_code == 400
        assert "metadata" in ssrf.json()["error"]["message"]

        missing = await client.post("/debug/backends", json={"id": "x"})
        assert missing.status_code == 400

        ok = await client.post(
            "/debug/backends", json={"id": "extra", "base_url": "http://127.0.0.1:19012"}
        )
        assert ok.status_code == 200
        assert "extra" in [b.id for b in gw.pool.list()]


def test_config_file_with_bad_url_fails_loudly(tmp_path):
    cfg = tmp_path / "dio.yaml"
    cfg.write_text(
        "backends:\n  - id: evil\n    url: file:///etc/passwd\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError) as excinfo:
        load_config_file(str(cfg))
    assert "http://" in str(excinfo.value)


# --------------------------------------------------------- TGI token counting
def test_tgi_translation_reports_prompt_tokens():
    out = tgi_generate_to_openai_chat({"generated_text": "hello there"}, model="m", prompt_tokens=11)
    usage = out["usage"]
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] >= 1
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_tgi_translation_prefers_server_reported_generated_tokens():
    out = tgi_generate_to_openai_chat(
        {"generated_text": "hi", "details": {"generated_tokens": 42}},
        model="m",
        prompt_tokens=7,
    )
    assert out["usage"] == {"prompt_tokens": 7, "completion_tokens": 42, "total_tokens": 49}


def test_tgi_translation_handles_list_payload():
    out = tgi_generate_to_openai_chat(
        [{"generated_text": "ok", "details": {"generated_tokens": 3}}], model="m", prompt_tokens=2
    )
    assert out["usage"]["total_tokens"] == 5


# -------------------------------------------------------------- Retry-After
@pytest.mark.asyncio
async def test_retry_after_matches_health_probe_horizon():
    gw = DIOGateway(
        backends=[Backend(id="b0", base_url="http://127.0.0.1:19011")],
        slo_ms=5000,
        health_interval_s=7.0,
    )
    # Exactly what the health loop does when the only engine stops answering.
    gw.scheduler.set_healthy("b0", False)
    async with _client(gw) as client:
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 503
        assert r.headers["retry-after"] == "7"


# ------------------------------------------------- Ollama routing headers
@pytest.mark.asyncio
async def test_ollama_nonstream_reports_serving_engine():
    from dio.backends import MockBackendServer

    srv = MockBackendServer(port=19911, latency_mult=1.0, decode_ms_per_token=1.0, name="gpu-x")
    await srv.start()
    try:
        gw = DIOGateway(
            backends=[Backend(id="gpu-x", base_url=srv.base_url)], admission_off=True
        )
        async with _client(gw) as client:
            chat = await client.post(
                "/api/chat",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
            assert chat.status_code == 200
            assert chat.headers.get("x-dio-backend") == "gpu-x"

            gen = await client.post(
                "/api/generate", json={"model": "m", "prompt": "hi", "stream": False}
            )
            assert gen.status_code == 200
            assert gen.headers.get("x-dio-backend") == "gpu-x"
    finally:
        await srv.stop()
