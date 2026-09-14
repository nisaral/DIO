"""Unit tests for multi-model routing across heterogeneous backends."""

import pytest
from dio.backends import Backend
from dio.gateway import DIOGateway
from dio.scheduler import AdmissionError, Scheduler


def test_resolve_model_backends():
    model_map = {
        "meta-llama/Llama-3-8B-Instruct": ["gpu0"],
        "mistralai/Mistral-7B-Instruct-v0.2": ["gpu1"],
        "codellama/CodeLlama-34b-Instruct": ["gpu0", "gpu1"],
    }
    gw = DIOGateway(backends=[], model_map=model_map)

    # Exact match
    assert gw._resolve_model_backends("meta-llama/Llama-3-8B-Instruct") == ["gpu0"]
    # Partial case-insensitive match
    assert gw._resolve_model_backends("mistral") == ["gpu1"]
    assert gw._resolve_model_backends("codellama") == ["gpu0", "gpu1"]
    # Generic default returns None (all backends eligible)
    assert gw._resolve_model_backends("default") is None
    assert gw._resolve_model_backends("") is None
    # Unmatched model returns empty list -> rejected
    assert gw._resolve_model_backends("gpt-4-turbo") == []


def test_scheduler_respects_allowed_backends():
    s = Scheduler(strategy="nlms", slo_ms=1e9, admission_off=True)
    s.register("b0", tier="small")
    s.register("b1", tier="small")
    s.register("b2", tier="small")

    # Constrain to b1
    wid, dec = s.pick("hello world", allowed_backends=["b1"])
    assert wid == "b1"

    # Constrain to b0 or b2
    wid, dec = s.pick("hello world", allowed_backends=["b0", "b2"])
    assert wid in ("b0", "b2")

    # Constrain to empty list -> should raise AdmissionError
    with pytest.raises(AdmissionError):
        s.pick("hello world", allowed_backends=[])


def test_gateway_models_endpoint_includes_model_map():
    from fastapi.testclient import TestClient

    b0 = Backend(id="b0", base_url="http://127.0.0.1:9001", tier="small")
    gw = DIOGateway(
        backends=[b0],
        model_map={"llama3": ["b0"], "mistral": ["b0"]},
    )
    client = TestClient(gw.app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json().get("data", [])
    model_ids = {m["id"] for m in data}
    assert "llama3" in model_ids
    assert "mistral" in model_ids
