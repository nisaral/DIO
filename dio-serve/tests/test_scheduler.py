"""Unit tests — no GPU / network required."""

from dio.scheduler import AblationFlags, AdmissionError, Scheduler


def test_nlms_learns_slope():
    s = Scheduler(strategy="nlms", dual=True, admission_off=True, slo_ms=1e9)
    s.register("w0", tier="small")
    for i in range(30):
        tokens = 50 + i
        # true: 2ms/token + 40ms
        actual = 2.0 * tokens + 40.0
        s.pick("x" * (tokens * 4), tokens=tokens)
        s.feedback("w0", actual, tokens)
    snap = s.predictors["w0"].snapshot()
    assert snap["updates"] == 30
    assert 1.0 < snap["fast_slope"] < 3.5


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
