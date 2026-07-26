"""Unit tests for vLLM /metrics parsing and hybrid cost (A/B/C)."""

from dio.engine_metrics import (
    parse_prometheus_text,
    prefix_hit_rate_delta,
    snapshot_from_metrics,
)
from dio.scheduler import AdmissionError, Scheduler


SAMPLE_METRICS = """
# HELP vllm:gpu_cache_usage_perc GPU KV cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.72
# HELP vllm:num_requests_waiting Waiting requests
vllm:num_requests_waiting 4.0
vllm:num_requests_running 2.0
vllm:prefix_cache_hits 80.0
vllm:prefix_cache_queries 100.0
"""


def test_parse_prometheus_text():
    m = parse_prometheus_text(SAMPLE_METRICS)
    assert abs(m["vllm:gpu_cache_usage_perc"] - 0.72) < 1e-9
    assert m["vllm:num_requests_waiting"] == 4.0


def test_snapshot_from_metrics():
    snap = snapshot_from_metrics(parse_prometheus_text(SAMPLE_METRICS))
    assert snap.ok
    assert abs(snap.kv_cache_usage - 0.72) < 1e-9
    assert snap.num_waiting == 4.0
    assert abs(snap.prefix_hit_rate - 0.8) < 1e-9


def test_snapshot_hit_rate_metric_variant():
    text = "vllm:gpu_prefix_cache_hit_rate 0.55\nvllm:kv_cache_usage_perc 0.1\n"
    snap = snapshot_from_metrics(parse_prometheus_text(text))
    assert abs(snap.prefix_hit_rate - 0.55) < 1e-9
    assert abs(snap.kv_cache_usage - 0.1) < 1e-9


def test_prefix_hit_rate_delta():
    a = snapshot_from_metrics({"vllm:prefix_cache_hits": 10, "vllm:prefix_cache_queries": 20})
    b = snapshot_from_metrics({"vllm:prefix_cache_hits": 18, "vllm:prefix_cache_queries": 30})
    dh, dq, rate = prefix_hit_rate_delta(a, b)
    assert dh == 8.0 and dq == 10.0 and abs(rate - 0.8) < 1e-9


def test_hybrid_cost_prefers_low_kv():
    s = Scheduler(
        strategy="nlms",
        admission_off=True,
        use_engine_metrics=True,
        kv_cache_cost_ms=1000.0,
        engine_queue_cost_ms=100.0,
        cache_bonus_ms=0.0,
    )
    s.register("full", total_vram_mb=24000, free_vram_mb=24000)
    s.register("empty", total_vram_mb=24000, free_vram_mb=24000)
    # Equal NLMS priors
    s.set_engine_metrics(
        "full",
        {"ok": True, "kv_cache_usage": 0.9, "num_waiting": 5, "prefix_hit_rate": 0.0},
    )
    s.set_engine_metrics(
        "empty",
        {"ok": True, "kv_cache_usage": 0.1, "num_waiting": 0, "prefix_hit_rate": 0.0},
    )
    wid, dec = s.pick("hello world", tokens=50)
    assert wid == "empty"
    assert dec.engine_kv_cost_ms > 0 or dec.engine_queue_cost_ms >= 0


def test_affinity_sticky_routing():
    s = Scheduler(
        strategy="nlms",
        admission_off=True,
        use_engine_metrics=False,
        cache_bonus_ms=5000.0,  # large bonus so affinity wins
    )
    s.register("a")
    s.register("b")
    # Make a slightly slower so without affinity we'd pick b
    s.predictors["a"].fast_slope = 3.0
    s.predictors["a"].slow_slope = 3.0
    s.predictors["b"].fast_slope = 1.0
    s.predictors["b"].slow_slope = 1.0
    prompt = "SESSION_PREFIX_abc " + ("x" * 40)
    # First pick lands on faster b
    w1, d1 = s.pick(prompt, tokens=20)
    assert w1 == "b"
    assert not d1.affinity_hit
    # Second pick with same prefix: affinity should stick to b even if we slow b
    s.predictors["b"].fast_slope = 10.0
    s.predictors["b"].slow_slope = 10.0
    w2, d2 = s.pick(prompt, tokens=20)
    assert w2 == "b"
    assert d2.affinity_hit
    m = s.metrics()["affinity"]
    assert m["hits"] >= 1
    assert m["decisions"] >= 2


def test_admission_empirical_warmup_not_absolute():
    """Empirical mode must not reject on cold absolute ŷ alone."""
    s = Scheduler(
        strategy="nlms",
        admission_mode="empirical",
        slo_ms=50.0,
        recent_latency_window=64,
    )
    s.register("w0")
    s.predictors["w0"].fast_slope = 50.0
    s.predictors["w0"].slow_slope = 50.0
    s.predictors["w0"].intercept = 1000.0
    # Warmup: absolute would reject; empirical should admit
    wid, _ = s.pick("hello " * 30, tokens=100)
    assert wid == "w0"
    assert s.admission.would_reject_absolute >= 1
    assert s.admission.rejected_slo == 0
    assert s.admission.absolute_vs_active_disagree >= 1


def test_admission_empirical_rejects_when_tail_hot():
    s = Scheduler(
        strategy="nlms",
        admission_mode="empirical",
        admission_percentile=95.0,
        slo_ms=100.0,
        recent_latency_window=32,
    )
    s.register("w0")
    # Fill observed latency above SLO
    for _ in range(20):
        s.recent_e2e.append(500.0)
    s.predictors["w0"].fast_slope = 1.0
    s.predictors["w0"].slow_slope = 1.0
    s.predictors["w0"].intercept = 50.0
    try:
        s.pick("hi", tokens=10)
        # May or may not reject depending on score quartile with single worker
        # Single worker: worst_quartile True → should reject
        assert False, "expected AdmissionError"
    except AdmissionError:
        assert s.admission.rejected_slo >= 1


def test_metrics_exposes_hybrid_and_affinity():
    s = Scheduler(strategy="nlms", admission_off=True, use_engine_metrics=True)
    s.register("w0")
    s.set_engine_metrics(
        "w0",
        {"ok": True, "kv_cache_usage": 0.3, "num_waiting": 1, "prefix_hit_rate": 0.5},
    )
    s.pick("p", tokens=5)
    m = s.metrics()
    assert m["use_engine_metrics"] is True
    assert "w0" in m["engine_metrics"]
    assert "affinity" in m
