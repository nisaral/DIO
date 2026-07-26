"""
DIO — Distributed Inference Orchestrator
========================================

Non-invasive control plane that sits in front of vLLM / SGLang / TGI / Ollama
(any OpenAI-compatible HTTP server) and routes requests with:

  • Dual-timescale NLMS latency learning (online, O(1) updates)
  • Hybrid joint cost: NLMS ranking + optional vLLM /metrics (KV, queue, prefix)
  • Session/prefix affinity for multi-turn workloads
  • Admission decoupled from absolute ŷ (empirical / rank_only / absolute)

Quick start
-----------
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
from dio.engine_metrics import EngineSnapshot, scrape_metrics_url
from dio.gateway import DIOGateway
from dio.scheduler import (
    AblationFlags,
    AdmissionStats,
    DualTimescaleNLMS,
    RoutingDecision,
    Scheduler,
)

__version__ = "0.2.1"
__all__ = [
    "Backend",
    "BackendPool",
    "DIOConfig",
    "DIOGateway",
    "Scheduler",
    "DualTimescaleNLMS",
    "RoutingDecision",
    "AdmissionStats",
    "AblationFlags",
    "EngineSnapshot",
    "scrape_metrics_url",
    "__version__",
]
