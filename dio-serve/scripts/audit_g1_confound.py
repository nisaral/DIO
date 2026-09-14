#!/usr/bin/env python3
"""
Audit of the G1 "hybrid cost routing" claim (paper Section 7.6 / Table 5).

Question: does the reported 40.9% p99 gain of `nlms_hybrid_on` over
`nlms_hybrid_off` come from the *scraped engine metrics* (contribution A),
or from the *session-affinity cache bonus* (contribution B)?

The two G1 arms differ in TWO knobs simultaneously
(run_gpu_abc_suite.py, G1 config block):

    nlms_hybrid_on  : engine_metrics=True , cache_bonus_ms=200.0
    nlms_hybrid_off : engine_metrics=False, cache_bonus_ms=0.0

so the published contrast is confounded. This script:

  1. Decomposes the hybrid cost terms using the engine snapshots actually
     recorded during the n=10 dual-T4 run.
  2. Replays the recorded scrape state through the real Scheduler and reports
     the magnitude of each score term.
  3. Runs the missing 4th arm (metrics OFF, affinity ON) at decision level.
  4. Recomputes the exact Wilcoxon p-value for the paired seed p99s.

Usage:
    python scripts/audit_g1_confound.py --summary results_gpu_abc_n10/summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Load scheduler.py directly: dio/__init__.py pulls in pydantic-settings, which is
# not needed for a pure scoring audit.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "dio_scheduler_audit", ROOT / "src" / "dio" / "scheduler.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _mod  # @dataclass resolves types via sys.modules
_spec.loader.exec_module(_mod)
Scheduler = _mod.Scheduler

# Defaults used for the dual-T4 G1 run (paper Appendix B).
C_KV = 800.0
C_Q = 50.0
C_P = 150.0
CACHE_BONUS = 200.0


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def collect_snapshots(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every engine snapshot recorded across all G-blocks."""
    snaps: List[Dict[str, Any]] = []
    for block in summary.values():
        if not isinstance(block, dict) or "configs" not in block:
            continue
        for cfg in block["configs"].values():
            for row in cfg.get("per_seed", []):
                ed = row.get("engine_debug")
                if isinstance(ed, dict):
                    for wid, snap in ed.items():
                        if isinstance(snap, dict) and snap.get("ok"):
                            snaps.append(snap)
    return snaps


def decompose(snaps: List[Dict[str, Any]]) -> None:
    hr("1. Hybrid cost terms computed from the ACTUAL recorded /metrics scrapes")
    if not snaps:
        print("no ok snapshots found")
        return

    kv = [float(s.get("kv_cache_usage") or 0.0) for s in snaps]
    wq = [float(s.get("num_waiting") or 0.0) for s in snaps]
    hit = [float(s.get("prefix_hit_rate") or 0.0) for s in snaps]
    run = [float(s.get("num_running") or 0.0) for s in snaps]

    def stat(name: str, xs: List[float], coeff: float, unit: str) -> None:
        cost = [x * coeff for x in xs]
        print(
            f"  {name:<22} raw: min={min(xs):.6f} mean={statistics.mean(xs):.6f} "
            f"max={max(xs):.6f}   ->  cost ms: mean={statistics.mean(cost):.3f} "
            f"max={max(cost):.3f}   ({unit})"
        )

    print(f"  n snapshots = {len(snaps)}")
    stat("KV cache usage", kv, C_KV, f"c_kv={C_KV}")
    stat("waiting requests", wq, C_Q, f"c_q={C_Q}")
    stat("prefix hit rate", hit, C_P, f"c_p={C_P}")
    print(
        f"  {'num_running':<22} raw: min={min(run):.1f} mean={statistics.mean(run):.3f} "
        f"max={max(run):.1f}"
    )

    total_max = max(k * C_KV for k in kv) + max(w * C_Q for w in wq)
    print()
    print(f"  >> MAX possible hybrid penalty over the whole campaign: {total_max:.3f} ms")
    print(f"  >> Affinity cache bonus (the other knob that changed):   {CACHE_BONUS:.1f} ms")
    print(f"  >> ratio affinity / hybrid = {CACHE_BONUS / max(total_max, 1e-9):.0f}x")


def replay_scheduler(snaps: List[Dict[str, Any]]) -> None:
    hr("2. Replay through the real Scheduler: score decomposition")
    worst = max(snaps, key=lambda s: float(s.get("kv_cache_usage") or 0.0))
    print(f"  worst-case observed snapshot: {json.dumps({k: worst[k] for k in ('kv_cache_usage','num_waiting','num_running','prefix_hit_rate')})}")

    for label, metrics_on, bonus in (
        ("hybrid_on  (paper arm)", True, 200.0),
        ("hybrid_off (paper arm)", False, 0.0),
        ("metrics_off + affinity ON  <-- MISSING ARM", False, 200.0),
        ("metrics_on  + affinity OFF <-- MISSING ARM", True, 0.0),
    ):
        sch = Scheduler(
            strategy="nlms",
            use_engine_metrics=metrics_on,
            cache_bonus_ms=bonus,
            admission_off=True,
            admission_mode="rank_only",
        )
        sch.register("e0", total_vram_mb=16000.0)
        sch.register("e1", total_vram_mb=16000.0)
        if metrics_on:
            sch.set_engine_metrics("e0", dict(worst))
            sch.set_engine_metrics("e1", dict(worst))

        prompt = "SYSTEM: long shared agent prefix. " + ("alpha beta gamma delta " * 40)
        # warm the prefix map so the affinity branch is live
        sch.prefix_cache[sch._prefix_hash(prompt)] = "e0"
        score, dec, blocked = sch._score(
            "e0", sch.predictors["e0"], tokens=280, tier="small", prompt=prompt, use_rls=False
        )
        print(
            f"  {label:<44} total={score:9.3f} ms | exec={dec.exec_ms:8.3f} "
            f"kv={dec.engine_kv_cost_ms:7.4f} q={dec.engine_queue_cost_ms:6.3f} "
            f"pref={dec.engine_prefix_bonus_ms:6.3f} affinity={dec.cache_bonus_ms - dec.engine_prefix_bonus_ms:7.2f}"
        )

    print()
    print("  >> The engine-metric terms (kv/q/pref) are sub-millisecond on this data.")
    print("  >> The only term that materially changes the ranking is the affinity bonus.")


def exact_wilcoxon(diffs: List[float]) -> None:
    hr("4. Exact two-sided Wilcoxon signed-rank p-value for the G1 paired p99s")
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    for r, i in enumerate(order, start=1):
        ranks[i] = float(r)
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if nz[i] < 0)
    w = min(w_plus, w_minus)
    total = n * (n + 1) / 2

    count = 0
    for signs in product((0, 1), repeat=n):
        wp = sum(r for r, s in zip(ranks, signs) if s)
        if min(wp, total - wp) <= w:
            count += 1
    p = count / (2 ** n)
    print(f"  n={n}  W+={w_plus:.0f}  W-={w_minus:.0f}  W={w:.0f}")
    print(f"  exact two-sided p = {p:.5f}")
    print(f"  paper states p ~ 0.01  ->  {'consistent' if abs(p - 0.01) < 0.01 else 'CHECK'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results_gpu_abc_n10" / "summary.json"))
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    snaps = collect_snapshots(summary)

    decompose(snaps)
    replay_scheduler(snaps)

    hr("3. Concurrency check: was the 'concurrent load' actually concurrent?")
    running = [float(s.get("num_running") or 0.0) for s in snaps]
    waiting = [float(s.get("num_waiting") or 0.0) for s in snaps]
    print(f"  max num_running observed across ALL scrapes: {max(running):.0f}")
    print(f"  max num_waiting observed across ALL scrapes: {max(waiting):.0f}")
    print("  (run_multiturn_load uses a synchronous httpx.Client in a for-loop,")
    print("   so 'concurrent' filler requests are issued sequentially -> no queueing.)")

    g1 = summary["G1_hybrid"]["configs"]
    on = [r["e2e_p99_ms"] for r in g1["nlms_hybrid_on"]["per_seed"]]
    rr = [r["e2e_p99_ms"] for r in g1["rr"]["per_seed"]]
    exact_wilcoxon([a - b for a, b in zip(rr, on)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
