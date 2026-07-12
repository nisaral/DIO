"""
Dual-timescale NLMS + joint cost router + Roofline-style admission.

Ported from the Go control plane (DIO/internal/scheduler) so the Python package
is self-contained — no Go binary required for research or production wrap mode.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class AblationFlags:
    name: str = "full"
    disable_queue: bool = False
    disable_vram_soft: bool = False
    disable_vram_hard: bool = False
    disable_tier: bool = False
    disable_cache: bool = False
    single_timescale: bool = False


@dataclass
class RoutingDecision:
    worker_id: str
    exec_ms: float
    wait_ms: float
    tier_cost_ms: float
    vram_cost_ms: float
    cache_bonus_ms: float
    total_ms: float
    tokens: int
    strategy: str = "nlms"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "exec_ms": self.exec_ms,
            "wait_ms": self.wait_ms,
            "tier_cost_ms": self.tier_cost_ms,
            "vram_cost_ms": self.vram_cost_ms,
            "cache_bonus_ms": self.cache_bonus_ms,
            "total_ms": self.total_ms,
            "tokens": self.tokens,
            "strategy": self.strategy,
        }


@dataclass
class AdmissionStats:
    admitted: int = 0
    rejected_slo: int = 0
    rejected_vram: int = 0
    rejected_no_worker: int = 0
    completed_under_slo: int = 0
    completed_over_slo: int = 0
    completed_total: int = 0
    sum_e2e_ms: float = 0.0

    def snapshot(self, slo_ms: float, admission_enabled: bool) -> Dict[str, Any]:
        total = self.completed_total
        goodput_frac = (self.completed_under_slo / total) if total else 0.0
        avg = (self.sum_e2e_ms / total) if total else 0.0
        return {
            "admitted": self.admitted,
            "rejected_slo": self.rejected_slo,
            "rejected_vram": self.rejected_vram,
            "rejected_no_worker": self.rejected_no_worker,
            "completed_under_slo": self.completed_under_slo,
            "completed_over_slo": self.completed_over_slo,
            "completed_total": total,
            "goodput_fraction": goodput_frac,
            "avg_e2e_ms": avg,
            "slo_ms": slo_ms,
            "admission_enabled": admission_enabled,
        }


class DualTimescaleNLMS:
    """Per-backend latency model: y ≈ s * tokens + b."""

    def __init__(
        self,
        *,
        mu_fast: float = 0.1,
        mu_slow: float = 0.01,
        mu_bias: float = 0.005,
        blend: float = 0.8,
        initial_slope: float = 0.1,
        initial_intercept: float = 50.0,
        dual: bool = True,
        frozen: bool = False,
        tier: str = "small",
        total_vram_mb: float = 24000.0,
    ) -> None:
        self.mu_fast = mu_fast
        self.mu_slow = mu_slow
        self.mu_bias = mu_bias
        self.blend = blend
        self.fast_slope = initial_slope
        self.slow_slope = initial_slope
        self.intercept = initial_intercept
        self.avg_latency = 0.0
        self.dual = dual and not frozen
        self.frozen = frozen
        self.tier = tier
        self.total_vram_mb = total_vram_mb
        self.update_count = 0
        self.sum_abs = 0.0
        self.sum_rel = 0.0
        self.sum_sq = 0.0
        self.pending = 0
        self.free_vram_mb = total_vram_mb
        self.healthy = True
        self._lock = threading.Lock()

    def effective_slope(self) -> float:
        if self.dual:
            return self.blend * self.fast_slope + (1.0 - self.blend) * self.slow_slope
        return self.fast_slope

    def mode(self) -> str:
        if self.frozen:
            return "STATIC"
        return "DUAL" if self.dual else "SINGLE"

    def estimate(self, tokens: int) -> Tuple[float, float]:
        with self._lock:
            base = self.effective_slope() * max(1, tokens) + self.intercept
            return base, self.avg_latency if self.avg_latency > 0 else base

    def update(self, actual_ms: float, tokens: int) -> Dict[str, float]:
        tokens = max(1, tokens)
        with self._lock:
            pred = self.effective_slope() * tokens + self.intercept
            err = actual_ms - pred
            abs_err = abs(err)
            rel = abs_err / max(actual_ms, 1.0)
            self.sum_abs += abs_err
            self.sum_rel += rel
            self.sum_sq += err * err
            self.update_count += 1

            if not self.frozen:
                grad = err / float(tokens)
                self.fast_slope += self.mu_fast * grad
                if self.dual:
                    self.slow_slope += self.mu_slow * grad
                else:
                    self.slow_slope = self.fast_slope
                self.intercept += self.mu_bias * err
                self.fast_slope = max(0.1, self.fast_slope)
                self.slow_slope = max(0.1, self.slow_slope)
                self.intercept = max(0.1, self.intercept)

            if self.avg_latency <= 0:
                self.avg_latency = actual_ms
            else:
                self.avg_latency = 0.9 * self.avg_latency + 0.1 * actual_ms

            return {
                "predicted": pred,
                "actual": actual_ms,
                "abs_err": abs_err,
                "rel_err": rel,
                "fast_slope": self.fast_slope,
                "slow_slope": self.slow_slope,
            }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            n = max(1, self.update_count)
            return {
                "fast_slope": self.fast_slope,
                "slow_slope": self.slow_slope,
                "intercept": self.intercept,
                "avg_latency_ms": self.avg_latency,
                "updates": self.update_count,
                "mode": self.mode(),
                "mae_ms": self.sum_abs / n if self.update_count else 0.0,
                "mape_pct": (self.sum_rel / n) * 100.0 if self.update_count else 0.0,
                "pending": self.pending,
                "free_vram_mb": self.free_vram_mb,
                "tier": self.tier,
                "healthy": self.healthy,
                "frozen": self.frozen,
            }


class SimpleRLS:
    """2×2 RLS baseline (paper comparison)."""

    def __init__(self, lam: float = 0.99, slope: float = 0.1, intercept: float = 50.0) -> None:
        self.lam = lam
        self.slope = slope
        self.intercept = intercept
        self.P = [[1000.0, 0.0], [0.0, 1000.0]]
        self.avg_latency = 0.0
        self.update_count = 0
        self._lock = threading.Lock()

    def estimate(self, tokens: int) -> Tuple[float, float]:
        with self._lock:
            base = self.slope * max(1, tokens) + self.intercept
            return base, self.avg_latency if self.avg_latency > 0 else base

    def update(self, actual_ms: float, tokens: int) -> None:
        tokens = max(1, tokens)
        with self._lock:
            x0, x1 = float(tokens), 1.0
            pred = self.slope * x0 + self.intercept
            err = actual_ms - pred
            px0 = self.P[0][0] * x0 + self.P[0][1] * x1
            px1 = self.P[1][0] * x0 + self.P[1][1] * x1
            denom = max(1e-9, self.lam + x0 * px0 + x1 * px1)
            k0, k1 = px0 / denom, px1 / denom
            self.slope += k0 * err
            self.intercept += k1 * err
            p00, p01, p10, p11 = self.P[0][0], self.P[0][1], self.P[1][0], self.P[1][1]
            self.P[0][0] = (p00 - k0 * (x0 * p00 + x1 * p10)) / self.lam
            self.P[0][1] = (p01 - k0 * (x0 * p01 + x1 * p11)) / self.lam
            self.P[1][0] = (p10 - k1 * (x0 * p00 + x1 * p10)) / self.lam
            self.P[1][1] = (p11 - k1 * (x0 * p01 + x1 * p11)) / self.lam
            self.slope = max(0.1, self.slope)
            self.intercept = max(0.1, self.intercept)
            self.update_count += 1
            if self.avg_latency <= 0:
                self.avg_latency = actual_ms
            else:
                self.avg_latency = 0.9 * self.avg_latency + 0.1 * actual_ms


class AdmissionError(Exception):
    def __init__(self, message: str, retry_after_sec: int = 1) -> None:
        super().__init__(message)
        self.retry_after_sec = max(1, retry_after_sec)


class Scheduler:
    """
    Joint cost router.

    score = wait + predicted_exec + tier_penalty + vram_penalty - cache_bonus
    Admit only if min score ≤ SLO (unless admission_off).
    """

    def __init__(
        self,
        *,
        strategy: str = "nlms",
        dual: bool = True,
        ablation: Optional[AblationFlags] = None,
        slo_ms: float = 5000.0,
        admission_off: bool = False,
        batch_size: float = 8.0,
        tier_mismatch_ms: float = 500.0,
        cache_bonus_ms: float = 200.0,
        vram_soft_mb: float = 4096.0,
        vram_hard_mb: float = 2400.0,
        mu_fast: float = 0.1,
        mu_slow: float = 0.01,
        mu_bias: float = 0.005,
        blend: float = 0.8,
        initial_slope: float = 0.1,
        initial_intercept: float = 50.0,
        static_slope: float = 1.0,
        static_intercept: float = 50.0,
        decision_log_size: int = 200,
        pred_history_size: int = 5000,
    ) -> None:
        self.strategy = strategy.lower().replace("-", "_")
        self.dual = dual
        self.ablation = ablation or AblationFlags()
        if self.ablation.single_timescale:
            self.dual = False
        self.slo_ms = slo_ms
        self.admission_off = admission_off
        self.batch_size = batch_size
        self.tier_mismatch_ms = tier_mismatch_ms
        self.cache_bonus_ms = cache_bonus_ms
        self.vram_soft_mb = vram_soft_mb
        self.vram_hard_mb = vram_hard_mb
        self._mu = (mu_fast, mu_slow, mu_bias, blend, initial_slope, initial_intercept)
        self.static_slope = static_slope
        self.static_intercept = static_intercept

        self._lock = threading.Lock()
        self.predictors: Dict[str, DualTimescaleNLMS] = {}
        self.rls: Dict[str, SimpleRLS] = {}
        self.prefix_cache: Dict[int, str] = {}
        self.rr_index = 0
        # deques: O(1) append + auto-drop for tight control-plane loops
        self.decision_log: Deque[Dict[str, Any]] = deque(maxlen=decision_log_size)
        self.decision_log_size = decision_log_size
        self.pred_history: Deque[Dict[str, Any]] = deque(maxlen=pred_history_size)
        self.pred_history_size = pred_history_size
        self.admission = AdmissionStats()
        self.last_decision: Optional[RoutingDecision] = None

    def register(
        self,
        worker_id: str,
        *,
        tier: str = "small",
        total_vram_mb: float = 24000.0,
        free_vram_mb: Optional[float] = None,
    ) -> None:
        mu_fast, mu_slow, mu_bias, blend, init_s, init_b = self._mu
        frozen = self.strategy == "static"
        slope = self.static_slope if frozen else init_s
        intercept = self.static_intercept if frozen else init_b
        with self._lock:
            self.predictors[worker_id] = DualTimescaleNLMS(
                mu_fast=mu_fast,
                mu_slow=mu_slow,
                mu_bias=mu_bias,
                blend=blend,
                initial_slope=slope,
                initial_intercept=intercept,
                dual=self.dual and not frozen,
                frozen=frozen,
                tier=tier,
                total_vram_mb=total_vram_mb,
            )
            if free_vram_mb is not None:
                self.predictors[worker_id].free_vram_mb = free_vram_mb
            self.rls[worker_id] = SimpleRLS(slope=init_s, intercept=init_b)

    def unregister(self, worker_id: str) -> None:
        with self._lock:
            self.predictors.pop(worker_id, None)
            self.rls.pop(worker_id, None)

    def set_vram(self, worker_id: str, free_mb: float) -> None:
        with self._lock:
            if worker_id in self.predictors:
                self.predictors[worker_id].free_vram_mb = free_mb

    def set_healthy(self, worker_id: str, healthy: bool) -> None:
        with self._lock:
            if worker_id in self.predictors:
                self.predictors[worker_id].healthy = healthy

    def _prefix_hash(self, text: str) -> int:
        return hash(text[:100]) & 0xFFFFFFFF

    def _score(
        self,
        worker_id: str,
        pred: DualTimescaleNLMS,
        tokens: int,
        tier: str,
        prompt: str,
        use_rls: bool,
    ) -> Tuple[Optional[float], Optional[RoutingDecision], bool]:
        """Returns (score, breakdown, blocked)."""
        abl = self.ablation
        free = pred.free_vram_mb

        if not abl.disable_vram_hard:
            if free > 0 and free < self.vram_hard_mb and tokens > 1000:
                return None, None, True

        if not abl.disable_tier:
            if tier == "large" and pred.tier != "large":
                return None, None, True

        if use_rls:
            exec_ms, avg = self.rls[worker_id].estimate(tokens)
        else:
            exec_ms, avg = pred.estimate(tokens)
        if exec_ms >= 1e300 or math.isinf(exec_ms):
            return None, None, True

        wait = 0.0
        if not abl.disable_queue:
            avg_t = avg if avg > 0 else exec_ms
            wait = (pred.pending / self.batch_size) * avg_t

        tier_cost = 0.0
        if not abl.disable_tier and tier == "small" and pred.tier == "large":
            tier_cost = self.tier_mismatch_ms

        vram_cost = 0.0
        if not abl.disable_vram_soft and pred.total_vram_mb > 0 and free < self.vram_soft_mb:
            vram_cost = (1.0 - free / pred.total_vram_mb) * 1000.0

        cache_bonus = 0.0
        if not abl.disable_cache:
            h = self._prefix_hash(prompt)
            if self.prefix_cache.get(h) == worker_id:
                cache_bonus = self.cache_bonus_ms

        total = wait + exec_ms + tier_cost + vram_cost - cache_bonus
        dec = RoutingDecision(
            worker_id=worker_id,
            exec_ms=exec_ms,
            wait_ms=wait,
            tier_cost_ms=tier_cost,
            vram_cost_ms=vram_cost,
            cache_bonus_ms=cache_bonus,
            total_ms=total,
            tokens=tokens,
            strategy=self.strategy,
        )
        return total, dec, False

    def pick(
        self,
        prompt: str,
        *,
        tier: str = "small",
        tokens: Optional[int] = None,
    ) -> Tuple[str, RoutingDecision]:
        """
        Select backend. Raises AdmissionError if no safe worker.
        Formal rule: reject if no feasible worker or min_w S_w > SLO.
        """
        tokens = tokens if tokens is not None else max(1, len(prompt) // 4)
        with self._lock:
            ids = [wid for wid, p in self.predictors.items() if p.healthy]
            if not ids:
                self.admission.rejected_no_worker += 1
                raise AdmissionError("no healthy backends registered")

            if self.strategy in ("round_robin", "roundrobin"):
                self.rr_index = (self.rr_index + 1) % len(ids)
                wid = ids[self.rr_index]
                pred = self.predictors[wid]
                pred.pending += 1
                dec = RoutingDecision(wid, 0, 0, 0, 0, 0, 0, tokens, self.strategy)
                self.admission.admitted += 1
                self._log_decision(dec)
                return wid, dec

            if self.strategy in ("least_loaded", "leastloaded", "least_load"):
                wid = min(ids, key=lambda i: self.predictors[i].pending)
                self.predictors[wid].pending += 1
                dec = RoutingDecision(wid, 0, 0, 0, 0, 0, 0, tokens, self.strategy)
                self.admission.admitted += 1
                self._log_decision(dec)
                return wid, dec

            use_rls = self.strategy == "rls"
            best_id: Optional[str] = None
            best_score = float("inf")
            best_dec: Optional[RoutingDecision] = None
            any_candidate = False

            for wid in ids:
                score, dec, blocked = self._score(
                    wid, self.predictors[wid], tokens, tier, prompt, use_rls
                )
                if blocked or score is None or dec is None:
                    continue
                any_candidate = True
                if score < best_score:
                    best_score = score
                    best_id = wid
                    best_dec = dec

            if best_id is None or best_dec is None:
                if any_candidate:
                    self.admission.rejected_vram += 1
                    raise AdmissionError(
                        "all backends VRAM/tier blocked",
                        retry_after_sec=2,
                    )
                self.admission.rejected_no_worker += 1
                raise AdmissionError("no feasible backend")

            if not self.admission_off and best_score > self.slo_ms:
                self.admission.rejected_slo += 1
                retry = max(1, int(best_score / 1000.0))
                raise AdmissionError(
                    f"predicted latency {best_score:.0f}ms exceeds SLO {self.slo_ms:.0f}ms",
                    retry_after_sec=retry,
                )

            self.predictors[best_id].pending += 1
            self.prefix_cache[self._prefix_hash(prompt)] = best_id
            self.admission.admitted += 1
            self._log_decision(best_dec)
            self.last_decision = best_dec
            return best_id, best_dec

    def _log_decision(self, dec: RoutingDecision) -> None:
        self.decision_log.append(dec.as_dict())

    def release(self, worker_id: str) -> None:
        with self._lock:
            if worker_id in self.predictors and self.predictors[worker_id].pending > 0:
                self.predictors[worker_id].pending -= 1

    def feedback(
        self,
        worker_id: str,
        e2e_ms: float,
        tokens: int,
        *,
        predicted_ms: Optional[float] = None,
    ) -> None:
        with self._lock:
            if worker_id in self.predictors:
                info = self.predictors[worker_id].update(e2e_ms, tokens)
                if worker_id in self.rls:
                    self.rls[worker_id].update(e2e_ms, tokens)
                sample = {
                    "unix_ms": int(time.time() * 1000),
                    "worker_id": worker_id,
                    "tokens": tokens,
                    **info,
                    "mode": self.predictors[worker_id].mode(),
                }
                self.pred_history.append(sample)
            self.admission.completed_total += 1
            self.admission.sum_e2e_ms += e2e_ms
            if e2e_ms <= self.slo_ms:
                self.admission.completed_under_slo += 1
            else:
                self.admission.completed_over_slo += 1
            if worker_id in self.predictors and self.predictors[worker_id].pending > 0:
                self.predictors[worker_id].pending -= 1

    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            workers = {wid: p.snapshot() for wid, p in self.predictors.items()}
            n = len(self.pred_history)
            mae = sum(s["abs_err"] for s in self.pred_history) / n if n else 0.0
            mape = (sum(s["rel_err"] for s in self.pred_history) / n * 100.0) if n else 0.0
            tail = list(self.pred_history)[-200:] if n else []
            return {
                "strategy": self.strategy,
                "nlms_mode": "dual" if self.dual else "single",
                "ablation": self.ablation.name,
                "workers": workers,
                "admission": self.admission.snapshot(self.slo_ms, not self.admission_off),
                "last_decision": self.last_decision.as_dict() if self.last_decision else None,
                "decisions": list(self.decision_log),
                "prediction": {
                    "count": n,
                    "mae_ms": mae,
                    "mape_pct": mape,
                    "samples": tail,
                },
            }

    def reset_stats(self) -> None:
        with self._lock:
            self.admission = AdmissionStats()
            self.pred_history.clear()
            self.decision_log.clear()
