"""Runtime configuration (env + CLI)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DIOConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DIO_", extra="ignore")

    # Routing
    strategy: Literal["nlms", "rls", "static", "round_robin", "least_loaded"] = "nlms"
    nlms_mode: Literal["dual", "single"] = "dual"
    ablation: Literal[
        "full", "no_queue", "no_vram", "no_vram_hard", "no_tier", "no_cache", "no_dual"
    ] = "full"

    # Admission: see admission_mode. slo_ms is the absolute/empirical budget.
    slo_ms: float = 5000.0
    admission_off: bool = False
    # absolute = reject if min ŷ-cost > slo (legacy; ŷ MAPE-sensitive)
    # empirical = reject using rolling observed latency percentile (preferred)
    # rank_only = VRAM/tier hard blocks only; NLMS used purely for ranking
    admission_mode: Literal["absolute", "empirical", "rank_only"] = "empirical"
    admission_percentile: float = 95.0  # for empirical mode
    recent_latency_window: int = 64

    # Cost coefficients (paper defaults)
    tier_mismatch_ms: float = 500.0
    cache_bonus_ms: float = 200.0
    vram_soft_limit_mb: float = 4096.0
    vram_hard_limit_mb: float = 2400.0
    batch_size: float = 8.0

    # NLMS
    mu_fast: float = 0.1
    mu_slow: float = 0.01
    mu_bias: float = 0.005
    fast_slow_blend: float = 0.8
    # Mildly pessimistic cold-start for real engines (still online-adapted).
    initial_slope: float = 2.0
    initial_intercept: float = 150.0
    static_slope: float = 1.0
    static_intercept: float = 50.0

    # Token feature for NLMS (prefer HF tokenizer when available)
    tokenizer_name: Optional[str] = None  # e.g. Qwen/Qwen2.5-3B-Instruct
    use_tokenizer: bool = True

    # Proxy
    host: str = "0.0.0.0"
    port: int = 8085
    request_timeout_s: float = 300.0
    health_interval_s: float = 5.0

    # Observability
    log_decisions: bool = True
    decision_log_size: int = 200
    pred_history_size: int = 5000


def ablation_from_name(name: str) -> "AblationFlags":
    from dio.scheduler import AblationFlags

    n = (name or "full").lower()
    f = AblationFlags(name=n)
    if n in ("no_queue", "-queue"):
        f.disable_queue = True
    elif n in ("no_vram", "-vram"):
        f.disable_vram_soft = True
        f.disable_vram_hard = True
    elif n == "no_vram_hard":
        f.disable_vram_hard = True
    elif n in ("no_tier", "-tier", "-tiers"):
        f.disable_tier = True
    elif n in ("no_cache", "-cache"):
        f.disable_cache = True
    elif n in ("no_dual", "single", "single_mu"):
        f.single_timescale = True
    return f
