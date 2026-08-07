#!/usr/bin/env python3
"""
A/B/C offline experiment suite (library-first, no GPU required).

  A  Hybrid cost: NLMS ranking + injected engine metrics (KV / queue)
     vs NLMS-only vs Round-Robin under asymmetric engine pressure.
  B  Session/prefix affinity: multi-turn sticky routing hit-rate vs RR.
  C  Admission safety: absolute (trusts ŷ) vs empirical vs rank_only
     under intentionally mis-scaled ŷ (high MAPE regime).

Usage::

    cd dio-serve && pip install -e .
    python scripts/run_abc_experiments.py
    python scripts/run_abc_experiments.py --seeds 20 --out results_abc

Writes results_abc/summary.json + paper_snippets.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dio.scheduler import AdmissionError, Scheduler  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mean_std(xs: Sequence[float]) -> Dict[str, float]:
    xs = [float(x) for x in xs]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"mean": m, "std": sd, "n": len(xs)}


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[i]


# ---------------------------------------------------------------------------
# A — Hybrid cost under engine pressure asymmetry
# ---------------------------------------------------------------------------
def run_A_hybrid(seeds: int, n_req: int) -> Dict[str, Any]:
    """
    Two workers with equal true latency, but engine metrics differ:
      eng_hot: high KV + waiting queue  → should be avoided by hybrid
      eng_cool: low pressure
    True service time same; benefit is avoiding artificial queue backlog.
    """
    results: Dict[str, List[float]] = {
        "nlms_only_p99": [],
        "hybrid_p99": [],
        "rr_p99": [],
        "hybrid_frac_cool": [],
        "nlms_frac_cool": [],
        "rr_frac_cool": [],
    }

    for seed in range(seeds):
        rng = random.Random(seed + 17)

        def true_lat(tokens: int) -> float:
            return 2.0 * tokens + 100.0 + rng.gauss(0, 15)

        def simulate(strategy: str, hybrid: bool) -> Dict[str, float]:
            s = Scheduler(
                strategy=strategy,
                admission_off=True,
                use_engine_metrics=hybrid,
                kv_cache_cost_ms=800.0 if hybrid else 0.0,
                engine_queue_cost_ms=80.0 if hybrid else 0.0,
                cache_bonus_ms=0.0,
                engine_prefix_hit_bonus_ms=0.0,
            )
            s.register("cool")
            s.register("hot")
            # Inject engine pressure (as if scraped from /metrics)
            if hybrid:
                s.set_engine_metrics(
                    "cool",
                    {
                        "ok": True,
                        "kv_cache_usage": 0.15,
                        "num_waiting": 0,
                        "prefix_hit_rate": 0.0,
                    },
                )
                s.set_engine_metrics(
                    "hot",
                    {
                        "ok": True,
                        "kv_cache_usage": 0.85,
                        "num_waiting": 8,
                        "prefix_hit_rate": 0.0,
                    },
                )
            # Pending backlog on hot to create real queue cost for all modes
            pending_hot = 0
            latencies: List[float] = []
            cool_n = 0
            for i in range(n_req):
                tokens = 40 + (i % 30)
                prompt = f"req-{seed}-{i} " + ("x" * 20)
                # Refresh metrics periodically (scrape cadence)
                if hybrid and i % 5 == 0:
                    s.set_engine_metrics(
                        "hot",
                        {
                            "ok": True,
                            "kv_cache_usage": 0.8 + 0.05 * rng.random(),
                            "num_waiting": float(6 + pending_hot // 2),
                            "prefix_hit_rate": 0.0,
                        },
                    )
                wid, _ = s.pick(prompt, tokens=tokens)
                if wid == "cool":
                    cool_n += 1
                # Hot worker has extra queue delay from pending
                extra = 40.0 * s.predictors[wid].pending if wid == "hot" else 0.0
                y = true_lat(tokens) + extra
                latencies.append(y)
                s.feedback(wid, y, tokens)
            return {
                "p99": pct(latencies, 99),
                "frac_cool": cool_n / max(1, n_req),
            }

        h = simulate("nlms", hybrid=True)
        n = simulate("nlms", hybrid=False)
        r = simulate("round_robin", hybrid=False)
        results["hybrid_p99"].append(h["p99"])
        results["nlms_only_p99"].append(n["p99"])
        results["rr_p99"].append(r["p99"])
        results["hybrid_frac_cool"].append(h["frac_cool"])
        results["nlms_frac_cool"].append(n["frac_cool"])
        results["rr_frac_cool"].append(r["frac_cool"])

    out = {k: mean_std(v) for k, v in results.items()}
    # vs RR improvement for hybrid
    imp = []
    for i in range(len(results["hybrid_p99"])):
        rr = results["rr_p99"][i]
        hy = results["hybrid_p99"][i]
        if rr > 0:
            imp.append(100.0 * (rr - hy) / rr)
    out["hybrid_p99_vs_rr_pct"] = mean_std(imp)
    return out


# ---------------------------------------------------------------------------
# B — Prefix / session affinity
# ---------------------------------------------------------------------------
def run_B_affinity(seeds: int, n_sessions: int, turns: int) -> Dict[str, Any]:
    """
    Multi-turn sessions: each session reuses a long shared prefix.
    Affinity bonus should stick to first backend; RR scatters.
    Also track 'engine' prefix hits as a proxy: same worker → higher hit rate.
    """
    aff_hit_rates: List[float] = []
    rr_hit_rates: List[float] = []
    aff_stickiness: List[float] = []  # fraction of turns staying on session home
    rr_stickiness: List[float] = []
    aff_p99: List[float] = []
    rr_p99: List[float] = []

    for seed in range(seeds):
        rng = random.Random(seed + 99)

        def run_mode(use_affinity: bool, strategy: str) -> Dict[str, float]:
            s = Scheduler(
                strategy=strategy,
                admission_off=True,
                use_engine_metrics=True,
                cache_bonus_ms=400.0 if use_affinity else 0.0,
                engine_prefix_hit_bonus_ms=100.0 if use_affinity else 0.0,
                kv_cache_cost_ms=0.0,
                engine_queue_cost_ms=0.0,
            )
            s.register("e0")
            s.register("e1")
            # Mild service difference so pure NLMS without affinity can switch
            s.predictors["e0"].fast_slope = 2.0
            s.predictors["e0"].slow_slope = 2.0
            s.predictors["e1"].fast_slope = 2.2
            s.predictors["e1"].slow_slope = 2.2

            session_home: Dict[int, str] = {}
            sticky = 0
            total_turns = 0
            engine_hits = 0  # synthetic: hit if same worker as previous turn for session
            last_worker: Dict[int, str] = {}
            lats: List[float] = []

            for sess in range(n_sessions):
                prefix = f"SYSTEM: you are agent-{sess} " + ("CONTEXT " * 30)
                for t in range(turns):
                    prompt = prefix + f"\nuser turn {t}: question {rng.randint(0, 999)}"
                    tokens = 50 + t * 5
                    # Engine reports higher prefix hit when sticky (synthetic scrape)
                    for wid in ("e0", "e1"):
                        hit = 0.7 if last_worker.get(sess) == wid else 0.1
                        s.set_engine_metrics(
                            wid,
                            {
                                "ok": True,
                                "kv_cache_usage": 0.3,
                                "num_waiting": 0,
                                "prefix_hit_rate": hit,
                            },
                        )
                    wid, dec = s.pick(prompt, tokens=tokens)
                    if sess not in session_home:
                        session_home[sess] = wid
                    if wid == session_home[sess]:
                        sticky += 1
                    if last_worker.get(sess) == wid:
                        engine_hits += 1
                    last_worker[sess] = wid
                    total_turns += 1
                    # Cache hit → cheaper true latency
                    base = 2.0 * tokens + 80.0
                    if dec.affinity_hit or (
                        t > 0 and wid == session_home[sess]
                    ):
                        y = base * 0.7 + rng.gauss(0, 8)
                    else:
                        y = base + rng.gauss(0, 8)
                    lats.append(max(1.0, y))
                    s.feedback(wid, y, tokens)

            m = s.metrics()["affinity"]
            return {
                "affinity_hit_rate": m["hit_rate"] if use_affinity else 0.0,
                "stickiness": sticky / max(1, total_turns),
                "engine_proxy_hit": engine_hits / max(1, total_turns - n_sessions),
                "p99": pct(lats, 99),
            }

        a = run_mode(True, "nlms")
        r = run_mode(False, "round_robin")
        aff_hit_rates.append(a["affinity_hit_rate"])
        aff_stickiness.append(a["stickiness"])
        aff_p99.append(a["p99"])
        rr_hit_rates.append(r["engine_proxy_hit"])
        rr_stickiness.append(r["stickiness"])
        rr_p99.append(r["p99"])

    return {
        "nlms_affinity_hit_rate": mean_std(aff_hit_rates),
        "nlms_session_stickiness": mean_std(aff_stickiness),
        "nlms_p99": mean_std(aff_p99),
        "rr_proxy_hit_rate": mean_std(rr_hit_rates),
        "rr_session_stickiness": mean_std(rr_stickiness),
        "rr_p99": mean_std(rr_p99),
    }


# ---------------------------------------------------------------------------
# C — Admission safety under unreliable ŷ
# ---------------------------------------------------------------------------
def run_C_admission(seeds: int, n_req: int) -> Dict[str, Any]:
    """
    True latency is mild (~150ms) and under SLO=500.
    NLMS priors/slopes are deliberately wrong so absolute ŷ >> SLO.
    absolute mode over-rejects; empirical/rank_only do not thrash.
    """
    modes = ("absolute", "empirical", "rank_only")
    stats: Dict[str, Dict[str, List[float]]] = {
        m: {"reject_rate": [], "goodput": [], "completed": [], "abs_disagree": []}
        for m in modes
    }

    for seed in range(seeds):
        rng = random.Random(seed + 3)
        for mode in modes:
            s = Scheduler(
                strategy="nlms",
                admission_mode=mode,
                admission_off=False,
                slo_ms=500.0,
                recent_latency_window=32,
                admission_percentile=95.0,
            )
            s.register("w0")
            # Poison ŷ: huge slope so absolute score >> SLO
            s.predictors["w0"].fast_slope = 40.0
            s.predictors["w0"].slow_slope = 40.0
            s.predictors["w0"].intercept = 800.0

            admitted = 0
            rejected = 0
            under = 0
            for i in range(n_req):
                tokens = 20 + (i % 10)
                # True service is fine
                true_y = 120.0 + 1.5 * tokens + abs(rng.gauss(0, 20))
                try:
                    wid, _ = s.pick("prompt " + str(i), tokens=tokens)
                    admitted += 1
                    s.feedback(wid, true_y, tokens)
                    if true_y <= 500.0:
                        under += 1
                except AdmissionError:
                    rejected += 1
                    # still observe ground truth for empirical window if we want
                    # (rejected requests don't complete)

            total_dec = admitted + rejected
            stats[mode]["reject_rate"].append(rejected / max(1, total_dec))
            stats[mode]["completed"].append(float(admitted))
            stats[mode]["goodput"].append(under / max(1, admitted) if admitted else 0.0)
            snap = s.admission.snapshot(500.0, True)
            stats[mode]["abs_disagree"].append(float(snap["absolute_vs_active_disagree"]))

    out: Dict[str, Any] = {}
    for mode in modes:
        out[mode] = {k: mean_std(v) for k, v in stats[mode].items()}
    return out


def run_hybrid_coeff_sensitivity(seeds: int, n_req: int) -> Dict[str, Any]:
    """
    ±50% on each hybrid coefficient under injected engine pressure.
    Reports p99 and frac_cool vs default; library-only (not dual-T4 multi-turn).
    """
    base = {"kv": 800.0, "q": 50.0, "p": 150.0}
    variants = [("default", 1.0, 1.0, 1.0)]
    for key in ("kv", "q", "p"):
        for scale, tag in ((0.5, "m50"), (1.5, "p50")):
            scales = {"kv": 1.0, "q": 1.0, "p": 1.0}
            scales[key] = scale
            variants.append((f"{key}_{tag}", scales["kv"], scales["q"], scales["p"]))

    out: Dict[str, Any] = {}
    for name, sk, sq, sp in variants:
        p99s: List[float] = []
        fracs: List[float] = []
        for seed in range(seeds):
            rng = random.Random(seed + 41)

            def true_lat(tokens: int) -> float:
                return 2.0 * tokens + 100.0 + rng.gauss(0, 15)

            s = Scheduler(
                strategy="nlms",
                admission_off=True,
                use_engine_metrics=True,
                kv_cache_cost_ms=base["kv"] * sk,
                engine_queue_cost_ms=base["q"] * sq,
                engine_prefix_hit_bonus_ms=base["p"] * sp,
                cache_bonus_ms=0.0,
            )
            s.register("cool")
            s.register("hot")
            cool_n = 0
            lats: List[float] = []
            for i in range(n_req):
                tokens = 40 + (i % 30)
                prompt = f"sens-{seed}-{i} " + ("x" * 20)
                if i % 5 == 0:
                    s.set_engine_metrics(
                        "cool",
                        {"ok": True, "kv_cache_usage": 0.15, "num_waiting": 0, "prefix_hit_rate": 0.0},
                    )
                    s.set_engine_metrics(
                        "hot",
                        {
                            "ok": True,
                            "kv_cache_usage": 0.8 + 0.05 * rng.random(),
                            "num_waiting": float(6 + s.predictors["hot"].pending // 2),
                            "prefix_hit_rate": 0.0,
                        },
                    )
                wid, _ = s.pick(prompt, tokens=tokens)
                if wid == "cool":
                    cool_n += 1
                extra = 40.0 * s.predictors[wid].pending if wid == "hot" else 0.0
                y = true_lat(tokens) + extra
                lats.append(y)
                s.feedback(wid, y, tokens)
            p99s.append(pct(lats, 99))
            fracs.append(cool_n / max(1, n_req))
        out[name] = {
            "p99": mean_std(p99s),
            "frac_cool": mean_std(fracs),
            "scales": {"kv": sk, "q": sq, "p": sp},
        }
    # relative p99 swing vs default
    d0 = out["default"]["p99"]["mean"]
    swings = []
    for name, block in out.items():
        if name == "default":
            continue
        m = block["p99"]["mean"]
        if d0 > 0:
            swings.append(100.0 * abs(m - d0) / d0)
    out["max_rel_p99_swing_pct"] = max(swings) if swings else 0.0
    out["mean_rel_p99_swing_pct"] = statistics.mean(swings) if swings else 0.0
    return out


def write_snippets(summary: Dict[str, Any], path: Path) -> None:
    A = summary["A_hybrid"]
    B = summary["B_affinity"]
    C = summary["C_admission"]
    lines = [
        "# A/B/C paper snippets (library multi-seed)",
        "",
        "## A — Hybrid engine metrics",
        f"- Hybrid p99 vs RR: {A['hybrid_p99_vs_rr_pct']['mean']:.1f}% ± {A['hybrid_p99_vs_rr_pct']['std']:.1f}%",
        f"- Hybrid frac cool worker: {A['hybrid_frac_cool']['mean']:.2f} ± {A['hybrid_frac_cool']['std']:.2f}",
        f"- NLMS-only frac cool: {A['nlms_frac_cool']['mean']:.2f}",
        f"- RR frac cool: {A['rr_frac_cool']['mean']:.2f}",
        "",
        "## B — Prefix / session affinity",
        f"- NLMS affinity hit rate: {B['nlms_affinity_hit_rate']['mean']:.2f} ± {B['nlms_affinity_hit_rate']['std']:.2f}",
        f"- NLMS session stickiness: {B['nlms_session_stickiness']['mean']:.2f}",
        f"- RR stickiness: {B['rr_session_stickiness']['mean']:.2f}",
        f"- NLMS p99: {B['nlms_p99']['mean']:.0f} vs RR p99: {B['rr_p99']['mean']:.0f}",
        "",
        "## C — Admission safety",
        f"- absolute reject rate: {C['absolute']['reject_rate']['mean']:.2f}",
        f"- empirical reject rate: {C['empirical']['reject_rate']['mean']:.2f}",
        f"- rank_only reject rate: {C['rank_only']['reject_rate']['mean']:.2f}",
        f"- absolute completed: {C['absolute']['completed']['mean']:.0f}",
        f"- empirical completed: {C['empirical']['completed']['mean']:.0f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-req", type=int, default=120)
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--out", type=str, default=str(ROOT / "results_abc"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.seeds = 3
        args.n_req = 40
        args.sessions = 8
        args.turns = 3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"A hybrid seeds={args.seeds} n_req={args.n_req}")
    A = run_A_hybrid(args.seeds, args.n_req)
    log(f"  hybrid p99 vs RR: {A['hybrid_p99_vs_rr_pct']}")

    log(f"B affinity sessions={args.sessions} turns={args.turns}")
    B = run_B_affinity(args.seeds, args.sessions, args.turns)
    log(f"  affinity hit={B['nlms_affinity_hit_rate']} stick={B['nlms_session_stickiness']}")

    log(f"C admission seeds={args.seeds}")
    C = run_C_admission(args.seeds, args.n_req)
    log(f"  abs reject={C['absolute']['reject_rate']} emp={C['empirical']['reject_rate']}")

    log(f"Hybrid coeff ±50% sensitivity seeds={args.seeds}")
    S = run_hybrid_coeff_sensitivity(args.seeds, args.n_req)
    log(f"  max rel p99 swing vs default: {S.get('max_rel_p99_swing_pct'):.2f}%")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seeds": args.seeds,
            "n_req": args.n_req,
            "sessions": args.sessions,
            "turns": args.turns,
        },
        "A_hybrid": A,
        "B_affinity": B,
        "C_admission": C,
        "hybrid_coeff_sensitivity": S,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_snippets(summary, out_dir / "paper_snippets.md")
    log(f"Wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
