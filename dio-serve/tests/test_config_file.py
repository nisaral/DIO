"""Tests for YAML config-as-code parsing, discovery, and backend extraction."""

import os
import tempfile

import pytest

from dio.config_file import (
    _minimal_yaml_parse,
    _parse_backend,
    detect_local_backends,
    generate_detected_config,
    generate_example_config,
    load_config_file,
)


def test_minimal_yaml_parse():
    sample = """
    # Comments
    host: 127.0.0.1
    port: 8085
    enabled: true
    disabled: false
    models: [llama3, mistral, codellama]
    """
    data = _minimal_yaml_parse(sample)
    assert data["host"] == "127.0.0.1"
    assert data["port"] == 8085
    assert data["enabled"] is True
    assert data["disabled"] is False
    assert data["models"] == ["llama3", "mistral", "codellama"]


def test_parse_backend():
    raw = {
        "id": "gpu0",
        "url": "http://localhost:8000",
        "tier": "large",
        "engine": "vllm",
        "models": ["meta-llama/Llama-3-8B", "mistral-7b"],
        "vram": 24000,
    }
    b = _parse_backend(raw, 0)
    assert b.id == "gpu0"
    assert b.base_url == "http://localhost:8000"
    assert b.tier == "large"
    assert b.model == "meta-llama/Llama-3-8B"
    assert b.labels["engine"] == "vllm"
    assert "mistral-7b" in b.labels["models"]


def test_load_config_file_roundtrip():
    content = """
backends:
  - id: b1
    url: http://localhost:8000
    tier: large
    models: [model-a]
  - id: b2
    url: http://localhost:8001
    tier: small
    models: [model-b]

scheduler:
  strategy: rls
  slo_ms: 15000

admission:
  mode: empirical
  percentile: 90

server:
  host: 127.0.0.1
  port: 9090
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(content)
        f.flush()
        temp_path = f.name

    try:
        backends, cfg, model_map = load_config_file(temp_path)
        assert len(backends) == 2
        assert backends[0].id == "b1"
        assert backends[1].id == "b2"
        assert cfg.strategy == "rls"
        assert cfg.slo_ms == 15000.0
        assert cfg.admission_mode == "empirical"
        assert cfg.admission_percentile == 90.0
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9090
        assert model_map == {"model-a": ["b1"], "model-b": ["b2"]}
    finally:
        os.unlink(temp_path)


def test_generate_config_and_discovery():
    cfg_text = generate_example_config()
    assert "backends:" in cfg_text
    assert "scheduler:" in cfg_text
    assert "admission:" in cfg_text

    detected = generate_detected_config([
        {
            "id": "ollama-local",
            "url": "http://127.0.0.1:11434",
            "tier": "small",
            "engine": "ollama",
            "models": ["llama3:latest"],
        }
    ])
    assert "ollama-local" in detected
    assert "llama3:latest" in detected


@pytest.mark.asyncio
async def test_detect_local_backends_no_throw():
    # Calling on arbitrary environment without running servers should return gracefully
    res = await detect_local_backends(timeout=0.2)
    assert isinstance(res, list)
