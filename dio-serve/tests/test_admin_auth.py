"""Admin-surface (/debug/*) authentication guard.

DIO ships with no authentication and can re-point backends at arbitrary URLs, so
the admin surface must be gated whenever an API key is configured.
"""

from __future__ import annotations

import httpx
import pytest

from dio.backends import Backend
from dio.gateway import DIOGateway


def _gateway(**overrides) -> DIOGateway:
    return DIOGateway(
        backends=[Backend(id="b0", base_url="http://127.0.0.1:19011", tier="small")],
        admission_off=True,
        **overrides,
    )


def _client(gw: DIOGateway) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://test")


@pytest.mark.asyncio
async def test_debug_open_when_no_key_configured():
    gw = _gateway()
    async with _client(gw) as client:
        assert (await client.get("/debug/metrics")).status_code == 200
        assert (await client.get("/health")).status_code == 200


@pytest.mark.asyncio
async def test_debug_requires_key_when_configured():
    gw = _gateway(api_key="s3cret")
    async with _client(gw) as client:
        # No credentials
        unauth = await client.get("/debug/metrics")
        assert unauth.status_code == 401
        assert unauth.headers.get("www-authenticate") == "Bearer"
        assert unauth.json()["error"]["type"] == "dio_unauthorized"

        # Wrong credentials
        assert (
            await client.get("/debug/metrics", headers={"Authorization": "Bearer nope"})
        ).status_code == 401

        # Correct credentials, either supported header
        assert (
            await client.get("/debug/metrics", headers={"Authorization": "Bearer s3cret"})
        ).status_code == 200
        assert (
            await client.get("/debug/metrics", headers={"X-DIO-API-Key": "s3cret"})
        ).status_code == 200

        # State-mutating admin routes are guarded too
        assert (await client.post("/debug/reset_stats")).status_code == 401
        assert (await client.post("/debug/chaos/vram", params={"free_mb": 10})).status_code == 401
        assert (await client.post("/debug/backends", json={"id": "evil", "url": "http://evil"})).status_code == 401


@pytest.mark.asyncio
async def test_key_does_not_gate_inference_or_health_routes():
    gw = _gateway(api_key="s3cret")
    async with _client(gw) as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/v1/models")).status_code == 200


@pytest.mark.asyncio
async def test_protect_debug_can_be_disabled():
    gw = _gateway(api_key="s3cret", protect_debug=False)
    async with _client(gw) as client:
        assert (await client.get("/debug/metrics")).status_code == 200


def test_api_key_read_from_env(monkeypatch):
    from dio.config import DIOConfig

    monkeypatch.setenv("DIO_API_KEY", "from-env")
    assert DIOConfig().api_key == "from-env"
