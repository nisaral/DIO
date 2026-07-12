"""
DIO — Distributed Inference Orchestrator
========================================

Non-invasive control plane that sits in front of vLLM / SGLang / TGI / Ollama
(any OpenAI-compatible HTTP server) and routes requests with:

  • Dual-timescale NLMS latency learning (online, O(1) updates)
  • Joint cost: predicted latency + queue + tier + VRAM pressure
  • Roofline-inspired admission (reject when min cost > SLO)

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
from dio.gateway import DIOGateway
from dio.scheduler import (
    AblationFlags,
    AdmissionStats,
    DualTimescaleNLMS,
    RoutingDecision,
    Scheduler,
)

__version__ = "0.2.0"
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
    "__version__",
]
