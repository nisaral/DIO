"""
YAML config-as-code for DIO.

Lets users define their full stack in a single ``dio.yaml``::

    backends:
      - id: local-gpu
        url: http://localhost:8000
        tier: large
        models: [meta-llama/Llama-3.2-3B-Instruct]

      - id: ollama
        url: http://localhost:11434
        tier: small
        engine: ollama
        models: [llama3, codellama, mistral]

    scheduler:
      strategy: nlms
      nlms_mode: dual
      slo_ms: 30000

    admission:
      mode: empirical
      percentile: 95

Load with::

    from dio.config_file import load_config_file
    backends, config = load_config_file("dio.yaml")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dio.backends import Backend
from dio.config import DIOConfig

log = logging.getLogger("dio.config_file")


class ConfigError(ValueError):
    """dio.yaml is readable YAML but not a usable DIO configuration."""

# Supported file names (searched in order)
_CONFIG_NAMES = ["dio.yaml", "dio.yml", "dio.json", ".dio.yaml", ".dio.yml"]


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML or JSON config file."""
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        return json.loads(text)

    # Try PyYAML (preferred), fall back to simple parsing
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    # Minimal YAML-subset parser for simple configs (no nested objects).
    # It cannot represent the nested backend list that DIO actually needs, so
    # refuse rather than start up with a silently wrong (empty) topology.
    log.warning(
        "PyYAML not installed; using minimal parser. Install with: pip install pyyaml"
    )
    parsed = _minimal_yaml_parse(text)
    if "backends" not in parsed:
        raise ConfigError(
            "PyYAML is required to read this config; the built-in fallback parser "
            "cannot represent nested backend lists. Install it with: pip install pyyaml"
        )
    return parsed


def _minimal_yaml_parse(text: str) -> Dict[str, Any]:
    """Very basic YAML parser for simple flat configs. Use PyYAML for full support."""
    result: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                # Simple list: [a, b, c]
                items = [v.strip().strip("'\"") for v in val[1:-1].split(",")]
                result[key] = [i for i in items if i]
            elif val.lower() in ("true", "yes"):
                result[key] = True
            elif val.lower() in ("false", "no"):
                result[key] = False
            elif val.replace(".", "", 1).replace("-", "", 1).isdigit():
                result[key] = float(val) if "." in val else int(val)
            else:
                result[key] = val.strip("'\"") if val else None
    return result


def discover_config(start_dir: Optional[str] = None) -> Optional[Path]:
    """
    Search for a DIO config file starting from ``start_dir`` (default: cwd)
    and walking upward.

    Returns the first found config path, or None.
    """
    d = Path(start_dir or os.getcwd()).resolve()
    for _ in range(10):  # max 10 levels up
        for name in _CONFIG_NAMES:
            candidate = d / name
            if candidate.is_file():
                return candidate
        parent = d.parent
        if parent == d:
            break
        d = parent
    return None


def _parse_backend(raw: Dict[str, Any], index: int) -> Backend:
    """Parse a single backend entry from the YAML config."""
    bid = raw.get("id") or raw.get("name") or f"backend-{index}"
    url = raw.get("url") or raw.get("base_url") or raw.get("address", "")

    if not url:
        raise ValueError(f"Backend '{bid}' missing 'url' field")

    engine = (raw.get("engine") or raw.get("type") or "openai").lower()

    # Auto-detect API style from engine type
    api_style = "openai"
    if engine == "ollama":
        api_style = "openai"  # Ollama has OpenAI compat mode
    elif engine in ("tgi", "text-generation-inference"):
        api_style = "tgi_generate"

    # Parse models list (for multi-model routing)
    models = raw.get("models") or raw.get("model")
    if isinstance(models, str):
        models = [models]
    elif models is None:
        models = []

    return Backend(
        id=bid,
        base_url=url.rstrip("/"),
        tier=raw.get("tier", "small"),
        model=models[0] if models else None,  # Primary model
        total_vram_mb=float(raw.get("total_vram_mb") or raw.get("vram", 24000)),
        free_vram_mb=float(raw.get("free_vram_mb") or raw.get("vram", 24000)),
        api_style=api_style,
        api_key=raw.get("api_key"),
        health_path=raw.get("health_path", "/health"),
        metrics_path=raw.get("metrics_path", "/metrics"),
        labels={"engine": engine, "models": ",".join(models)},
    )


def load_config_file(
    path: Optional[str] = None,
) -> Tuple[List[Backend], DIOConfig, Dict[str, List[str]]]:
    """
    Load backends and config from a YAML file.

    Returns:
        (backends, dio_config, model_map)
        model_map: { "model_name": ["backend_id_1", "backend_id_2"] }
    """
    if path:
        p = Path(path)
    else:
        p = discover_config()
        if p is None:
            raise FileNotFoundError(
                "No dio.yaml found. Create one with: dio init"
            )

    log.info("Loading config from %s", p)
    data = _load_yaml(p)
    if not isinstance(data, dict):
        raise ConfigError(
            f"{p}: expected a mapping at the top level but found "
            f"{type(data).__name__}. See dio.example.yaml for the expected shape."
        )

    # Parse backends
    backends: List[Backend] = []
    model_map: Dict[str, List[str]] = {}

    raw_backends = data.get("backends") or data.get("backend") or []
    if raw_backends and not isinstance(raw_backends, (list, dict)):
        raise ConfigError(
            f"{p}: 'backends' must be a list of backend entries, "
            f"got {type(raw_backends).__name__}."
        )
    if isinstance(raw_backends, dict):
        # Support { id: url } shorthand
        raw_backends = [{"id": k, "url": v} for k, v in raw_backends.items()]

    for i, raw in enumerate(raw_backends):
        if isinstance(raw, str):
            # Plain URL string
            raw = {"url": raw, "id": f"b{i}"}
        try:
            b = _parse_backend(raw, i)
        except ValueError as e:
            # A typo'd URL is a config problem the user can fix, not a traceback.
            raise ConfigError(f"{p}: {e}") from None
        backends.append(b)

        # Build model → backend mapping
        models = raw.get("models") or raw.get("model")
        if isinstance(models, str):
            models = [models]
        for model_name in models or []:
            model_map.setdefault(model_name, []).append(b.id)

    # Parse scheduler config
    sched = data.get("scheduler") or {}
    admission = data.get("admission") or {}
    server = data.get("server") or {}

    config_kwargs: Dict[str, Any] = {}

    # Scheduler settings
    if sched.get("strategy"):
        config_kwargs["strategy"] = sched["strategy"]
    if sched.get("nlms_mode"):
        config_kwargs["nlms_mode"] = sched["nlms_mode"]
    if sched.get("ablation"):
        config_kwargs["ablation"] = sched["ablation"]
    if sched.get("cache_bonus_ms") is not None:
        config_kwargs["cache_bonus_ms"] = float(sched["cache_bonus_ms"])
    if sched.get("affinity_cache_size") is not None:
        config_kwargs["affinity_cache_size"] = int(sched["affinity_cache_size"])
    if sched.get("learner_robust") is not None:
        config_kwargs["learner_robust"] = bool(sched["learner_robust"])

    # Admission settings
    if admission.get("mode"):
        config_kwargs["admission_mode"] = admission["mode"]
    if admission.get("slo_ms") is not None:
        config_kwargs["slo_ms"] = float(admission["slo_ms"])
    elif sched.get("slo_ms") is not None:
        config_kwargs["slo_ms"] = float(sched["slo_ms"])
    if admission.get("percentile") is not None:
        config_kwargs["admission_percentile"] = float(admission["percentile"])
    if admission.get("off") or admission.get("disabled"):
        config_kwargs["admission_off"] = True

    # Engine metrics
    if sched.get("engine_metrics") is not None:
        config_kwargs["engine_metrics"] = bool(sched["engine_metrics"])

    # Server settings
    if server.get("host"):
        config_kwargs["host"] = server["host"]
    if server.get("port") is not None:
        config_kwargs["port"] = int(server["port"])
    if server.get("timeout") is not None:
        config_kwargs["request_timeout_s"] = float(server["timeout"])
    if server.get("body_size_cap_bytes") is not None:
        config_kwargs["body_size_cap_bytes"] = int(server["body_size_cap_bytes"])
    if server.get("prompt_chars_cap") is not None:
        config_kwargs["prompt_chars_cap"] = int(server["prompt_chars_cap"])

    # Security (see DIOConfig): DIO is a control plane, not an auth layer.
    security = data.get("security") or {}
    if security.get("api_key"):
        config_kwargs["api_key"] = security["api_key"]
    if security.get("protect_debug") is not None:
        config_kwargs["protect_debug"] = bool(security["protect_debug"])
    if security.get("data_plane_auth") is not None:
        config_kwargs["data_plane_auth"] = bool(security["data_plane_auth"])

    # Tokenizer
    if sched.get("tokenizer"):
        config_kwargs["tokenizer_name"] = sched["tokenizer"]
        config_kwargs["use_tokenizer"] = True

    cfg = DIOConfig(**config_kwargs)

    log.info(
        "Loaded %d backends, %d model bindings, strategy=%s",
        len(backends),
        sum(len(v) for v in model_map.values()),
        cfg.strategy,
    )

    return backends, cfg, model_map


def generate_example_config() -> str:
    """Generate an example dio.yaml config string."""
    return """\
# DIO Configuration - Distributed Inference Orchestrator
# Docs: https://github.com/nisaral/DIO

backends:
  # Local vLLM instance
  - id: local-gpu
    url: http://localhost:8000
    tier: large
    engine: vllm
    models:
      - meta-llama/Llama-3.2-3B-Instruct

  # Second GPU / remote server
  # - id: remote-a100
  #   url: http://10.0.1.5:8000
  #   tier: large
  #   engine: vllm
  #   vram: 81920
  #   models:
  #     - meta-llama/Llama-3.1-70B-Instruct

  # Ollama (local CPU/GPU)
  # - id: ollama
  #   url: http://localhost:11434
  #   tier: small
  #   engine: ollama
  #   models:
  #     - llama3
  #     - codellama
  #     - mistral

scheduler:
  strategy: nlms        # nlms | rls | ewma | static | round_robin | least_loaded
  nlms_mode: dual       # dual | single
  engine_metrics: true  # scrape vLLM /metrics for KV-cache-aware routing
  cache_bonus_ms: 200   # session affinity bonus for multi-turn
  affinity_cache_size: 2048  # how many session prefixes stay pinned in the LRU

admission:
  mode: empirical       # empirical | strict | rank_only | absolute
  slo_ms: 30000         # max acceptable e2e latency (ms)
  percentile: 95        # for empirical / strict mode

server:
  host: 0.0.0.0
  port: 8085
  timeout: 300          # request timeout (s)
"""


async def detect_local_backends(timeout: float = 0.8) -> List[Dict[str, Any]]:
    """
    Probe standard local ports for running inference engines (Ollama, vLLM, SGLang).
    Returns a list of dicts describing discovered backends and their loaded models.
    """
    import httpx

    discovered: List[Dict[str, Any]] = []

    # 1. Ollama (port 11434)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get("http://127.0.0.1:11434/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") or m.get("model") for m in data.get("models", [])]
                discovered.append({
                    "id": "ollama-local",
                    "url": "http://127.0.0.1:11434",
                    "tier": "small",
                    "engine": "ollama",
                    "models": [m for m in models if m] or ["llama3"],
                })
    except Exception:
        pass

    # 2. vLLM / OpenAI on port 8000
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get("http://127.0.0.1:8000/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", [])]
                discovered.append({
                    "id": "vllm-primary",
                    "url": "http://127.0.0.1:8000",
                    "tier": "large",
                    "engine": "vllm",
                    "models": [m for m in models if m] or ["default"],
                })
    except Exception:
        pass

    # 3. vLLM / OpenAI on port 8001 (secondary GPU)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get("http://127.0.0.1:8001/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", [])]
                discovered.append({
                    "id": "vllm-secondary",
                    "url": "http://127.0.0.1:8001",
                    "tier": "large",
                    "engine": "vllm",
                    "models": [m for m in models if m] or ["default"],
                })
    except Exception:
        pass

    # 4. SGLang on port 30000
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get("http://127.0.0.1:30000/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id") for m in data.get("data", [])]
                discovered.append({
                    "id": "sglang-local",
                    "url": "http://127.0.0.1:30000",
                    "tier": "large",
                    "engine": "sglang",
                    "models": [m for m in models if m] or ["default"],
                })
    except Exception:
        pass

    return discovered


def generate_detected_config(discovered: List[Dict[str, Any]]) -> str:
    """Generate a dio.yaml string from discovered backends."""
    if not discovered:
        return generate_example_config()

    lines = [
        "# DIO Configuration - Auto-discovered Local Backends",
        "# Generated by: dio init",
        "# Docs: https://github.com/nisaral/DIO",
        "",
        "backends:",
    ]
    for b in discovered:
        lines.append(f"  - id: {b['id']}")
        lines.append(f"    url: {b['url']}")
        lines.append(f"    tier: {b['tier']}")
        lines.append(f"    engine: {b['engine']}")
        if b.get("models"):
            lines.append("    models:")
            for m in b["models"]:
                lines.append(f"      - {m}")
        lines.append("")

    lines.extend([
        "scheduler:",
        "  strategy: nlms        # nlms | rls | ewma | static | round_robin | least_loaded",
        "  nlms_mode: dual       # dual | single",
        "  engine_metrics: true  # scrape vLLM /metrics for KV-cache-aware routing",
        "  cache_bonus_ms: 200   # session affinity bonus for multi-turn",
        "  affinity_cache_size: 2048  # pinned session prefixes in the LRU",
        "",
        "admission:",
        "  mode: empirical       # empirical | strict | rank_only | absolute",
        "  slo_ms: 30000         # max acceptable e2e latency (ms)",
        "  percentile: 95        # for empirical / strict mode",
        "",
        "server:",
        "  host: 0.0.0.0",
        "  port: 8085",
        "  timeout: 300          # request timeout (s)",
        "",
    ])
    return "\n".join(lines)

