"""Regression tests for the gaps that used to be published in the README.

Every test here is a counter-example that was true before the fix:

* an over-budget but unequal pool answered 200s and reported ``rejected_slo == 0``
* the affinity LRU had no eviction count, so a collapsing hit rate had no cause
* one burst could wreck the learned slope/intercept
* a request body and a prompt had no ceiling at all
* ``/v1/*`` and ``/api/*`` could not be authenticated even with a key configured
* ``/v1/models``, ``/api/tags`` and routing each had their own idea of the model
  list, and the Ollama digest changed on every restart
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from dio.backends import Backend, MockBackendServer
from dio.gateway import DIOGateway, _model_digest
from dio.scheduler import AdmissionError, Scheduler

PORT_MODELS = 19961
PORT_AUTH = 19962

HONEST_MS = 250.0
BURST_MS = 30_000.0


def _client(gw: DIOGateway) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gw.app), base_url="http://test"
    )


def _chat(model: str = "mock-model") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


# --------------------------------------------------------------------- admission
def _saturated_pool(mode: str) -> Scheduler:
    """Two backends, one 10x better than the other, observed tail over budget."""
    s = Scheduler(strategy="nlms", admission_mode=mode, slo_ms=100.0)
    for wid, slope in (("fast", 1.5), ("slow", 15.0)):
        s.register(wid)
        s.predictors[wid].fast_slope = slope
        s.predictors[wid].slow_slope = slope
        s.predictors[wid].intercept = 50.0
    for _ in range(10):
        s.feedback("fast", 900.0, 100)
        s.feedback("slow", 1900.0, 100)
    return s


def test_strict_admission_sheds_where_empirical_only_ranks():
    # The documented default: the tail is over budget, but one backend is so much
    # better than the other that admission admits anyway -- 200s that miss the SLO.
    empirical = _saturated_pool("empirical")
    wid, _dec = empirical.pick("hello", tokens=100)
    assert wid == "fast"
    assert empirical.admission.rejected_slo == 0
    assert empirical.admission.rejected_slo_suppressed >= 1
    assert empirical.metrics()["admission"]["empirical_p_ms"] > empirical.slo_ms

    # strict: same workload, same tail, but the SLO is a contract.
    strict = _saturated_pool("strict")
    with pytest.raises(AdmissionError) as exc:
        strict.pick("hello", tokens=100)
    assert "strict_p95" in str(exc.value)
    assert strict.admission.rejected_slo == 1


# ---------------------------------------------------------------------- affinity
def test_affinity_evictions_are_visible():
    s = Scheduler(strategy="nlms", admission_off=True, slo_ms=1e9, affinity_cache_size=16)
    s.register("w0")
    for i in range(40):
        s.pick(f"prompt-{i}", tokens=4)
    aff = s.metrics()["affinity"]
    assert aff["capacity"] == 16
    assert aff["cache_size"] == 16
    # 40 distinct prefixes through a 16-entry LRU.
    assert aff["evictions"] == 24
    s.reset_stats()
    assert s.metrics()["affinity"]["evictions"] == 0


# ----------------------------------------------------------------------- learner
def _learn(pairs, robust: bool, concurrency: int = 1, probe_tokens: int = 100):
    """Feed (tokens, actual_ms) pairs and report the fit plus its estimate.

    ``concurrency`` > 1 admits several requests on the backend before any of them
    reports back, which is what makes those samples queue-contended.
    """
    s = Scheduler(strategy="nlms", admission_off=True, slo_ms=1e9, learner_robust=robust)
    s.register("w0")
    batch = []
    for tokens, actual in pairs:
        s.pick("x", tokens=tokens)
        batch.append((tokens, actual))
        if len(batch) >= concurrency:
            for t, a in batch:
                s.feedback("w0", a, t)
            batch = []
    for t, a in batch:
        s.feedback("w0", a, t)
    snap = s.predictors["w0"].snapshot()
    pred, _avg = s.predictors["w0"].estimate(probe_tokens)
    return snap, pred


def test_a_contended_burst_cannot_wreck_the_learner():
    # 8 requests in flight, alternating a normal engine with a queue-inflated one:
    # the shape the dogfooding swarm produced. The old update chased the queue into
    # the service-time model -- the slope ran to its ceiling and a normal request
    # was predicted orders of magnitude too slow -- so the router then avoided a
    # healthy backend for a reason that was not real.
    pattern = [(100, HONEST_MS), (100, BURST_MS)] * 15
    robust_snap, robust_pred = _learn(pattern, robust=True, concurrency=8)
    naive_snap, naive_pred = _learn(pattern, robust=False, concurrency=8)

    assert naive_snap["fast_slope"] > 100.0
    assert robust_snap["fast_slope"] * 10 < naive_snap["fast_slope"]
    assert robust_pred * 10 < naive_pred
    assert robust_pred < 3 * HONEST_MS

    # Clipping is reported rather than hidden, and mae/mape still count the raw
    # error, so the outlier stays visible in the aggregates.
    assert robust_snap["clipped_updates"] > 0
    assert naive_snap["clipped_updates"] == 0
    assert robust_snap["mae_ms"] > 1000.0


def test_an_uncontended_slowdown_is_believed_at_full_speed():
    # The guard must not make the router blind to a backend that really did get
    # slower (a thermal throttle, a co-tenant). With nothing queued the sample is
    # service time, so it gets the unguarded update -- this is the case the first
    # version of the fix got wrong, and the offline demo caught it.
    pairs = [(100, HONEST_MS)] * 30 + [(100, 8 * HONEST_MS)] * 20
    robust_snap, robust_pred = _learn(pairs, robust=True)
    _naive_snap, naive_pred = _learn(pairs, robust=False)
    assert robust_snap["clipped_updates"] == 0
    assert robust_pred == pytest.approx(naive_pred)
    assert robust_pred > 2 * HONEST_MS


def test_robust_updates_keep_the_documented_convergence():
    # Same workload as test_scheduler.test_nlms_learns_slope, robust update on.
    snap, _ = _learn(
        [(40 + (i % 40), 4.0 * (40 + (i % 40)) + 120.0) for i in range(60)],
        robust=True,
        probe_tokens=40,
    )
    assert 2.5 < snap["fast_slope"] < 6.0


# ------------------------------------------------------------------------- limits
@pytest.mark.asyncio
async def test_oversized_body_is_rejected_before_parsing():
    gw = DIOGateway(backends=[], body_size_cap_bytes=1024)
    async with _client(gw) as client:
        big = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x" * 4096}]},
        )
        small = await client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert big.status_code == 413
    assert big.json()["error"]["code"] == "request_too_large"
    # The cap is a size gate, not a shape gate: the small body still reaches the
    # ordinary validation path (503 here -- there are no backends registered).
    assert small.status_code != 413


@pytest.mark.asyncio
async def test_prompt_cap_uses_each_dialect_envelope():
    gw = DIOGateway(
        backends=[Backend(id="b0", base_url="http://127.0.0.1:19999", tier="small")],
        prompt_chars_cap=32,
    )
    async with _client(gw) as client:
        openai = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "y" * 100}]},
        )
        ollama = await client.post(
            "/api/chat",
            json={"model": "m", "messages": [{"role": "user", "content": "y" * 100}]},
        )
    assert openai.status_code == 413
    assert openai.json()["error"]["code"] == "request_too_large"
    assert "exceeds the configured cap" in openai.json()["error"]["message"]
    # Ollama clients read a top-level string, not an error object.
    assert ollama.status_code == 413
    assert isinstance(ollama.json()["error"], str)


# --------------------------------------------------------------------- data plane
@pytest.mark.asyncio
async def test_data_plane_auth_is_opt_in():
    srv = MockBackendServer(port=PORT_AUTH, latency_mult=0.01, decode_ms_per_token=0.01)
    await srv.start()
    try:
        backends = [Backend(id="b0", base_url=srv.base_url, tier="small", model="mock-model")]
        # A key on its own only guards /debug/*: inference stays open, which is
        # the documented control-plane behavior.
        open_gw = DIOGateway(backends=backends, api_key="s3cret", admission_off=True)
        async with _client(open_gw) as client:
            assert (await client.post("/v1/chat/completions", json=_chat())).status_code == 200

        guarded = DIOGateway(
            backends=backends, api_key="s3cret", data_plane_auth=True, admission_off=True
        )
        async with _client(guarded) as client:
            denied = await client.post("/v1/chat/completions", json=_chat())
            assert denied.status_code == 401
            assert denied.json()["error"]["type"] == "dio_unauthorized"
            assert denied.headers.get("www-authenticate") == "Bearer"
            allowed = await client.post(
                "/v1/chat/completions",
                json=_chat(),
                headers={"Authorization": "Bearer s3cret"},
            )
            assert allowed.status_code == 200
            # Probes stay open: a load balancer has no key.
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/debug/metrics")).status_code == 401
    finally:
        await srv.stop()


# ------------------------------------------------------------------ model surface
@pytest.mark.asyncio
async def test_model_listings_agree_and_are_routable():
    srv = MockBackendServer(port=PORT_MODELS, latency_mult=0.01, decode_ms_per_token=0.01)
    await srv.start()
    try:
        gw = DIOGateway(
            backends=[
                Backend(id="b0", base_url=srv.base_url, tier="small", model="engine-model")
            ],
            model_map={"declared-model": ["b0"]},
            admission_off=True,
        )
        async with _client(gw) as client:
            data = (await client.get("/v1/models")).json()["data"]
            tags = (await client.get("/api/tags")).json()["models"]
            served = await client.post(
                "/api/chat",
                json={"model": "engine-model", "messages": [{"role": "user", "content": "hi"}],
                      "stream": False},
            )
            unknown = await client.post(
                "/api/chat",
                json={"model": "nowhere", "messages": [{"role": "user", "content": "hi"}],
                      "stream": False},
            )
    finally:
        await srv.stop()

    openai_ids = {m["id"] for m in data}
    ollama_ids = {m["name"] for m in tags}
    assert openai_ids == ollama_ids
    # declared in config, declared on the backend, and probed from the engine.
    assert {"declared-model", "engine-model", "mock-model"} <= openai_ids
    # Every advertised name routes; anything else fails closed instead of being
    # forwarded to whichever engine happens to be first.
    assert served.status_code == 200
    assert unknown.status_code == 404


@pytest.mark.asyncio
async def test_version_has_one_source_of_truth():
    # dio/__init__.py used to hardcode its own copy of the version, so
    # /api/version (which imports it from there) disagreed with the FastAPI app
    # version and the package metadata, both of which read _version.py.
    import dio
    from dio._version import __version__

    gw = DIOGateway(backends=[])
    async with _client(gw) as client:
        reported = (await client.get("/api/version")).json()["version"]
    assert reported == dio.__version__ == gw.app.version == __version__


def test_model_digest_is_stable_across_processes():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from dio.gateway import _model_digest; print(_model_digest('llama3'))"
    )

    def digest_in_fresh_process(seed: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=str(root), capture_output=True, text=True, env=env
        )
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    # hash("llama3") differs per process; the tag listing must not.
    assert digest_in_fresh_process("1") == digest_in_fresh_process("2") == _model_digest("llama3")


# -------------------------------------------------------------------------- perf
def test_scoring_happens_once_per_backend_per_request(monkeypatch):
    from dio import scheduler as scheduler_mod

    scored: list[str] = []
    original = scheduler_mod.Scheduler._score

    def counting(self, worker_id, *args, **kwargs):
        scored.append(worker_id)
        return original(self, worker_id, *args, **kwargs)

    monkeypatch.setattr(scheduler_mod.Scheduler, "_score", counting)

    s = Scheduler(strategy="nlms", admission_off=True, slo_ms=1e9)
    for wid in ("a", "b", "c"):
        s.register(wid)
    s.pick("hello", tokens=10)
    s.pick("hello", tokens=10)
    # Two passes per request used to mean 12 evaluations for 3 backends.
    assert scored == ["a", "b", "c"] * 2
