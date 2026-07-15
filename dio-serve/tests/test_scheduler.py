"""Unit tests — no GPU / network required."""

from dio.scheduler import AblationFlags, AdmissionError, Scheduler


def test_nlms_learns_slope():
    s = Scheduler(strategy="nlms", dual=True, admission_off=True, slo_ms=1e9)
    s.register("w0", tier="small")
    # Ground truth near real-engine e2e scale (joint slope+intercept learning
    # needs enough samples when cold-start intercept is ~150 ms).
    for i in range(60):
        tokens = 40 + (i % 40)
        actual = 4.0 * tokens + 120.0
        s.pick("x" * (tokens * 4), tokens=tokens)
        s.feedback("w0", actual, tokens)
    snap = s.predictors["w0"].snapshot()
    assert snap["updates"] == 60
    assert 2.5 < snap["fast_slope"] < 6.0


def test_admission_rejects_over_slo():
    s = Scheduler(strategy="nlms", dual=True, admission_off=False, slo_ms=100.0)
    s.register("w0")
    s.predictors["w0"].fast_slope = 50.0
    s.predictors["w0"].slow_slope = 50.0
    s.predictors["w0"].intercept = 1000.0
    try:
        s.pick("hello world " * 20, tokens=100)
        assert False, "expected AdmissionError"
    except AdmissionError as e:
        assert e.retry_after_sec >= 1
        assert s.admission.rejected_slo >= 1


def test_round_robin_cycles():
    s = Scheduler(strategy="round_robin", admission_off=True)
    s.register("a")
    s.register("b")
    seen = [s.pick("hi")[0] for _ in range(4)]
    assert set(seen) == {"a", "b"}


def test_ablation_no_queue():
    s = Scheduler(
        strategy="nlms",
        ablation=AblationFlags(name="no_queue", disable_queue=True),
        admission_off=True,
        slo_ms=1e9,
    )
    s.register("w")
    s.predictors["w"].pending = 50
    _, dec = s.pick("hi", tokens=10)
    assert dec.wait_ms == 0.0
