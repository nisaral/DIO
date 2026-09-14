"""
DIO — Distributed Inference Orchestrator
========================================

Non-invasive control plane that sits in front of vLLM / SGLang / TGI / Ollama
(any OpenAI-compatible HTTP server) and routes requests with:

  • Dual-timescale NLMS latency learning (online, O(1) updates)
  • Hybrid joint cost: NLMS ranking + optional vLLM /metrics (KV, queue, prefix)
  • Session/prefix affinity for multi-turn workloads
  • Admission decoupled from absolute ŷ (empirical / rank_only / absolute)
  • Multi-model routing: dio.yaml maps model names → backend pools
  • Config-as-code: dio.yaml for one-file stack definition

Quick start
-----------
CLI (config file, auto-discovered)::

    dio init
    # edit dio.yaml
    dio serve

CLI (wrap two already-running vLLM servers)::

    pip install -e .
    dio serve --backend http://127.0.0.1:8000 --backend http://127.0.0.1:8001

Python API::

    from dio import DIOGateway, Backend
    gw = DIOGateway(backends=[
        Backend(id="gpu0", base_url="http://127.0.0.1:8000"),
        Backend(id="gpu1", base_url="http://127.0.0.1:8001", tier="large"),
    ])
    gw.run(host="0.0.0.0", port=8085)
"""

from dio.backends import Backend, BackendPool
from dio.config import DIOConfig
from dio.config_file import (
    ConfigError,
    detect_local_backends,
    discover_config,
    generate_detected_config,
    generate_example_config,
    load_config_file,
)
from dio.engine_metrics import EngineSnapshot, scrape_metrics_url
from dio.gateway import DIOGateway
from dio.mcp import DIOMCPServer
from dio.scheduler import (
    AblationFlags,
    AdmissionStats,
    DualTimescaleNLMS,
    RoutingDecision,
    Scheduler,
)

__version__ = "0.4.0"
__all__ = [
    "AblationFlags",
    "AdmissionStats",
    "Backend",
    "BackendPool",
    "ConfigError",
    "DIOConfig",
    "DIOGateway",
    "DIOMCPServer",
    "DualTimescaleNLMS",
    "EngineSnapshot",
    "RoutingDecision",
    "Scheduler",
    "__version__",
    "detect_local_backends",
    "discover_config",
    "generate_detected_config",
    "generate_example_config",
    "load_config_file",
    "scrape_metrics_url",
]

