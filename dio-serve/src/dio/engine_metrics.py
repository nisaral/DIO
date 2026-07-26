"""
Scrape OpenAI-compatible engine Prometheus /metrics (vLLM-style).

Still non-invasive: HTTP GET only, no engine source patches.
Handles metric name variants across vLLM versions:
  - vllm:gpu_cache_usage_perc | vllm:kv_cache_usage_perc
  - vllm:num_requests_waiting | vllm:num_requests_running
  - vllm:prefix_cache_hits / vllm:prefix_cache_queries
  - vllm:gpu_prefix_cache_hit_rate (older)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import httpx

log = logging.getLogger("dio.engine_metrics")

# Prometheus sample: name{labels} value  OR  name value
_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)"  # metric name
    r"(?:\{[^}]*\})?"  # optional labels
    r"\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$"
)


@dataclass
class EngineSnapshot:
    """Ground-truth engine state from /metrics (or empty if unavailable)."""

    kv_cache_usage: float = 0.0  # 0..1
    num_waiting: float = 0.0
    num_running: float = 0.0
    prefix_hit_rate: float = 0.0  # 0..1
    prefix_hits: float = 0.0
    prefix_queries: float = 0.0
    scraped_at: float = 0.0
    ok: bool = False
    source: str = "none"
    raw_keys: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float | str | bool]:
        return {
            "kv_cache_usage": self.kv_cache_usage,
            "num_waiting": self.num_waiting,
            "num_running": self.num_running,
            "prefix_hit_rate": self.prefix_hit_rate,
            "prefix_hits": self.prefix_hits,
            "prefix_queries": self.prefix_queries,
            "scraped_at": self.scraped_at,
            "ok": self.ok,
            "source": self.source,
        }


def parse_prometheus_text(text: str) -> Dict[str, float]:
    """Parse Prometheus exposition format into name -> last value (labels ignored)."""
    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        name, val = m.group(1), float(m.group(2))
        # keep last sample if multiple label sets
        out[name] = val
    return out


def _first(metrics: Dict[str, float], *names: str) -> Optional[float]:
    for n in names:
        if n in metrics:
            return metrics[n]
        # also try without vllm: prefix variants
        bare = n.split(":", 1)[-1]
        for k, v in metrics.items():
            if k.endswith(bare) or k == bare:
                return v
    return None


def snapshot_from_metrics(metrics: Dict[str, float]) -> EngineSnapshot:
    snap = EngineSnapshot(raw_keys=dict(metrics), scraped_at=time.time())
    if not metrics:
        return snap

    kv = _first(
        metrics,
        "vllm:gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc",
        "gpu_cache_usage_perc",
        "kv_cache_usage_perc",
    )
    if kv is not None:
        snap.kv_cache_usage = max(0.0, min(1.0, float(kv)))

    waiting = _first(metrics, "vllm:num_requests_waiting", "num_requests_waiting")
    if waiting is not None:
        snap.num_waiting = max(0.0, float(waiting))

    running = _first(metrics, "vllm:num_requests_running", "num_requests_running")
    if running is not None:
        snap.num_running = max(0.0, float(running))

    hit_rate = _first(
        metrics,
        "vllm:gpu_prefix_cache_hit_rate",
        "vllm:prefix_cache_hit_rate",
        "gpu_prefix_cache_hit_rate",
    )
    hits = _first(metrics, "vllm:prefix_cache_hits", "prefix_cache_hits")
    queries = _first(metrics, "vllm:prefix_cache_queries", "prefix_cache_queries")
    if hits is not None:
        snap.prefix_hits = float(hits)
    if queries is not None:
        snap.prefix_queries = float(queries)
    if hit_rate is not None:
        snap.prefix_hit_rate = max(0.0, min(1.0, float(hit_rate)))
    elif snap.prefix_queries > 0:
        snap.prefix_hit_rate = max(0.0, min(1.0, snap.prefix_hits / snap.prefix_queries))

    snap.ok = True
    snap.source = "prometheus"
    return snap


def scrape_metrics_url(
    url: str,
    *,
    timeout_s: float = 2.0,
    client: Optional[httpx.Client] = None,
) -> EngineSnapshot:
    """Synchronous scrape (used by background thread / scripts)."""
    own = client is None
    c = client or httpx.Client(timeout=timeout_s)
    try:
        r = c.get(url)
        if r.status_code >= 400:
            log.debug("metrics %s -> %s", url, r.status_code)
            return EngineSnapshot(scraped_at=time.time(), source=f"http_{r.status_code}")
        return snapshot_from_metrics(parse_prometheus_text(r.text))
    except Exception as e:
        log.debug("metrics scrape failed %s: %s", url, e)
        return EngineSnapshot(scraped_at=time.time(), source=f"error:{type(e).__name__}")
    finally:
        if own:
            c.close()


def metrics_url_for_backend(base_url: str, metrics_path: str = "/metrics") -> str:
    return base_url.rstrip("/") + (
        metrics_path if metrics_path.startswith("/") else f"/{metrics_path}"
    )


def prefix_hit_rate_delta(
    before: EngineSnapshot, after: EngineSnapshot
) -> Tuple[float, float, float]:
    """
    Return (delta_hits, delta_queries, interval_hit_rate) between two scrapes.
    """
    dh = max(0.0, after.prefix_hits - before.prefix_hits)
    dq = max(0.0, after.prefix_queries - before.prefix_queries)
    rate = (dh / dq) if dq > 0 else 0.0
    return dh, dq, rate
