"""Request-shape validation, error envelopes, and the empirical admission counter.

All three came out of letting a swarm of agents hammer a live gateway:

  * ``max_tokens: 10**9`` was forwarded unbounded and held a worker until the
    client gave up, and negative/absent ``messages`` were accepted;
  * framework errors (404/405/422) came back as FastAPI ``{"detail": ...}``, which
    an OpenAI SDK cannot turn into a useful exception;
  * ``rejected_slo`` stayed at 0 while the observed p95 exceeded the SLO, which
    made the admission gate look inert rather than deliberate.
"""

from __future__ import annotations

import httpx
import pytest

from dio.backends import Backend
from dio.gateway import DIOGateway
from dio.scheduler import Scheduler


def _gateway(**overrides) -> DIOGateway:
    # The backend is deliberately unreachable: every assertion below is about a
    # request that must be rejected *before* any upstream call is attempted.
    return DIOGateway(
        backends=[Backend(id="b0", base_url="http://127.0.0.1:19999")],
        admission_off=True,
        **overrides,
    )


def _client(gw: DIOGateway) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://test")


def _chat(**extra):
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(extra)
    return payload


# ------------------------------------------------------------ request shapes
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"model": "m"},  # messages missing
        {"model": "m", "messages": []},  # messages empty
        _chat(max_tokens=0),
        _chat(max_tokens=-5),
        _chat(max_tokens=1.5),
        _chat(max_tokens=True),
    ],
)
async def test_chat_rejects_abusive_or_malformed_bodies(payload):
    gw = _gateway()
    async with _client(gw) as client:
        r = await client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "invalid_request_error"
        assert r.json()["error"]["message"]


@pytest.mark.asyncio
async def test_max_tokens_cap_is_opt_in():
    payload = _chat(max_tokens=10_000_000)
    async with _client(_gateway()) as client:
        # Default: the engine's own max_model_len is authoritative, so no cap.
        assert (await client.post("/v1/chat/completions", json=payload)).status_code != 400
    async with _client(_gateway(max_tokens_cap=4096)) as client:
        r = await client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 400
        assert "cap of 4096" in r.json()["error"]["message"]


@pytest.mark.asyncio
async def test_completions_requires_a_prompt():
    gw = _gateway()
    async with _client(gw) as client:
        assert (await client.post("/v1/completions", json={"model": "m"})).status_code == 400
        assert (
            await client.post("/v1/completions", json={"model": "m", "prompt": "  "})
        ).status_code == 400


@pytest.mark.asyncio
async def test_ollama_routes_use_ollama_error_shape():
    gw = _gateway()
    async with _client(gw) as client:
        r = await client.post("/api/chat", json={"model": "m", "messages": []})
        assert r.status_code == 400
        assert isinstance(r.json()["error"], str)  # Ollama's shape, not OpenAI's


# ------------------------------------------------------------- error envelope
@pytest.mark.asyncio
async def test_framework_errors_are_openai_shaped():
    gw = _gateway()
    async with _client(gw) as client:
        not_found = await client.get("/v1/nope")
        assert not_found.status_code == 404
        assert not_found.json()["error"]["message"]

        wrong_method = await client.get("/v1/chat/completions")
        assert wrong_method.status_code == 405
        assert "error" in wrong_method.json()

        bad_json = await client.post(
            "/v1/chat/completions",
            content=b"{oops",
            headers={"Content-Type": "application/json"},
        )
        assert bad_json.status_code == 400
        assert bad_json.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_ollama_framework_errors_use_ollama_shape():
    gw = _gateway()
    async with _client(gw) as client:
        r = await client.get("/api/nope")
        assert r.status_code == 404
        assert r.json() == {"error": "Not Found"}


# --------------------------------------------------------------- /health view
@pytest.mark.asyncio
async def test_health_reports_the_slo_view():
    gw = _gateway()
    async with _client(gw) as client:
        body = (await client.get("/health")).json()
        assert body["status"] == "ok"
        assert body["slo"]["slo_ms"] == gw.config.slo_ms
        assert "goodput_fraction" in body["slo"]
        assert "rejected_slo_suppressed" in body["slo"]


# ------------------------------------------------- empirical admission counter
def test_empirical_gate_counts_suppressed_rejections():
    sched = Scheduler(strategy="nlms", slo_ms=100.0, admission_mode="empirical")
    sched.register("a")
    sched.register("b")
    sched.recent_e2e.extend([500.0] * 20)  # observed tail far above the SLO

    # A clearly better candidate exists -> admitting is deliberate, and visible.
    reject, reason = sched._should_reject_slo(best_score=50.0, scores=[50.0, 400.0])
    assert reject is False
    assert reason == "empirical_ok_better_backend_available"
    assert sched.metrics()["admission"]["rejected_slo_suppressed"] == 1

    # Uniformly saturated field -> the gate does trip.
    reject_uniform, _ = sched._should_reject_slo(best_score=100.0, scores=[100.0, 100.0])
    assert reject_uniform is True


def test_empirical_gate_rejects_when_only_one_backend_is_feasible():
    sched = Scheduler(strategy="nlms", slo_ms=100.0, admission_mode="empirical")
    sched.register("a")
    sched.recent_e2e.extend([500.0] * 20)
    reject, reason = sched._should_reject_slo(best_score=500.0, scores=[500.0])
    assert reject is True
    assert "SLO" in reason
