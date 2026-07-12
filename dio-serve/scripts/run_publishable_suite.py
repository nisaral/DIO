#!/usr/bin/env python3
"""
DIO Publishable Novelty Suite (library-first, runs anywhere)
===========================================================

Uses the installed ``dio`` package only — no Go, no Locust, no GPU required
for the default path.

Covers paper-critical claims:
  L0  Library API smoke + pick overhead microbench
  N1  Dual-timescale vs single-µ under burst jitter + thermal drift (MAPE)
  N2  Admission ON vs OFF under overload (goodput)
  N3  Multi-tier joint cost routing distribution
  N4  Ablations + baselines (RR, LL, RLS, STATIC, NLMS, −queue/−vram/−tier/−dual)
  N5  Control-plane scale (many backends, O(|W|) scan stays light)
  G1  Optional end-to-end gateway + mock HTTP backends (needs free ports)

Usage::

    cd dio-serve
    pip install -e .
    python scripts/run_publishable_suite.py
    python scripts/run_publishable_suite.py --quick
    python scripts/run_publishable_suite.py --skip-gateway   # pure scheduler only
    python scripts/run_publishable_suite.py --backends http://127.0.0.1:8000,http://127.0.0.1:8001

Outputs::

    results_publishable/summary.json
    results_publishable/tables.json
    results_publishable/figures/*.png   (if matplotlib installed)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import library under test
# ---------------------------------------------------------------------------
try:
    from dio import AblationFlags, Backend, DIOGateway, Scheduler
    from dio.backends import MockBackendServer
    from dio.scheduler import AdmissionError
except ImportError as e:
    print("ERROR: install package first:  cd dio-serve && pip install -e .", file=sys.stderr)
    print(e, file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_publishable"
FIGS = OUT / "figures"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[i]


# ---------------------------------------------------------------------------
# Simulated heterogeneous "engines" (no HTTP) — pure library path
# ---------------------------------------------------------------------------
@dataclass
class SimBackend:
    id: str
    slope: float
    bias: float
    tier: str = "small"
    free_vram: float = 20000.0
    total_vram: float = 24000.0
    jitter: float = 0.08
    thermal_start: int = 10**9
    thermal_ramp: float = 0.0
    thermal_max: float = 1.0
    n_served: int = 0

    def serve(self, tokens: int) -> float:
        self.n_served += 1
        th = 1.0
        if self.n_served >= self.thermal_start:
            th = min(self.thermal_max, 1.0 + (self.n_served - self.thermal_start) * self.thermal_ramp)
        noise = 1.0 + random.uniform(-self.jitter, self.jitter)
        return max(1.0, (self.bias + self.slope * tokens) * th * noise)


def simulate_load(
    sched: Scheduler,
    backends: Dict[str, SimBackend],
    *,
    n_requests: int,
    tokens_fn,
    tier_fn=lambda i: "small",
    open_loop: bool = False,
) -> Dict[str, Any]:
    """Drive Scheduler.pick/feedback against simulated latencies."""
    for b in backends.values():
        if b.id not in sched.predictors:
            sched.register(b.id, tier=b.tier, total_vram_mb=b.total_vram, free_vram_mb=b.free_vram)
        else:
            sched.set_vram(b.id, b.free_vram)

    lats: List[float] = []
    ok = fail = rej = 0
    routes: Dict[str, int] = {}
    t0 = time.perf_counter()

    for i in range(n_requests):
        tokens = int(tokens_fn(i))
        tier = tier_fn(i)
        prompt = "x" * max(4, tokens * 4)
        try:
            wid, dec = sched.pick(prompt, tier=tier, tokens=tokens)
        except AdmissionError:
            rej += 1
            fail += 1
            continue
        actual = backends[wid].serve(tokens)
        # keep scheduler VRAM view in sync if sim changes free_vram
        sched.set_vram(wid, backends[wid].free_vram)
        sched.feedback(wid, actual, tokens)
        lats.append(actual)
        ok += 1
        routes[wid] = routes.get(wid, 0) + 1

    wall = time.perf_counter() - t0
    adm = sched.metrics()["admission"]
    pred = sched.metrics()["prediction"]
    under = float(adm.get("completed_under_slo") or 0)
    return {
        "n_ok": ok,
        "n_fail": fail,
        "n_rejected": rej,
        "p50_ms": pct(lats, 50),
        "p95_ms": pct(lats, 95),
        "p99_ms": pct(lats, 99),
        "mean_ms": statistics.mean(lats) if lats else None,
        "wall_s": wall,
        "rps": ok / wall if wall > 0 else 0.0,
        "goodput_under_slo_rps": under / wall if wall > 0 else 0.0,
        "routes": routes,
        "admission": adm,
        "prediction": {"count": pred.get("count"), "mae_ms": pred.get("mae_ms"), "mape_pct": pred.get("mape_pct")},
        "workers": {k: v for k, v in sched.metrics()["workers"].items()},
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def exp_L0_library_overhead(n: int = 20000) -> Dict[str, Any]:
    log("=== L0 Library smoke + pick overhead ===")
    s = Scheduler(strategy="nlms", dual=True, admission_off=True, slo_ms=1e9)
    for i in range(8):
        s.register(f"w{i}")
    # warm
    for _ in range(500):
        w, _ = s.pick("warm", tokens=16)
        s.feedback(w, 80.0, 16)
    t0 = time.perf_counter()
    for i in range(n):
        w, _ = s.pick("hello world prompt", tokens=20)
        s.feedback(w, 90.0 + (i % 7), 20)
    us = (time.perf_counter() - t0) / n * 1e6
    log(f"  pick+feedback: {us:.1f} µs/req  (n={n})")
    assert us < 2000.0, "scheduler too slow for control plane"
    return {
        "name": "L0_overhead",
        "pick_feedback_us": us,
        "backends": 8,
        "iterations": n,
        "pass": us < 500.0,  # soft target: <0.5ms
    }


def exp_N1_dual_vs_single(n_req: int) -> List[Dict[str, Any]]:
    log("=== N1 Dual vs SINGLE (burst jitter + thermal drift) ===")
    results = []
    for dual, tag in [(True, "DUAL"), (False, "SINGLE")]:
        s = Scheduler(
            strategy="nlms",
            dual=dual,
            ablation=AblationFlags(name="full" if dual else "no_dual", single_timescale=not dual),
            admission_off=True,
            slo_ms=1e9,
        )
        backends = {
            "fast": SimBackend("fast", slope=10.0, bias=50.0, jitter=0.28),  # bursty
            "slow": SimBackend(
                "slow",
                slope=12.0,
                bias=80.0,
                jitter=0.05,
                thermal_start=25,
                thermal_ramp=0.03,
                thermal_max=2.0,
            ),
        }
        out = simulate_load(
            s,
            backends,
            n_requests=n_req,
            tokens_fn=lambda i: 40 + (i % 40) * 3,
        )
        out["name"] = f"N1_{tag}"
        out["nlms_mode"] = tag
        results.append(out)
        log(f"  {tag}: p99={out['p99_ms']:.1f} mape={out['prediction']['mape_pct']:.2f}% routes={out['routes']}")
    return results


def exp_N2_admission(n_req: int) -> List[Dict[str, Any]]:
    log("=== N2 Admission goodput (ON vs OFF) ===")
    results = []
    # Short requests ~ OK under SLO; long requests blow budget.
    # Admission ON should reject long ones → higher fraction of *completions* under SLO.
    slo = 700.0
    for admit_off, tag in [(True, "admission_off"), (False, "admission_on")]:
        s = Scheduler(
            strategy="nlms",
            dual=True,
            admission_off=admit_off,
            slo_ms=slo,
        )
        backends = {
            "a": SimBackend("a", slope=5.0, bias=80.0, jitter=0.03),
            "b": SimBackend("b", slope=6.0, bias=90.0, jitter=0.03),
        }
        # Mix: 50% short (tokens=40 → ~280ms), 50% long (tokens=200 → ~1080ms)
        out = simulate_load(
            s,
            backends,
            n_requests=n_req,
            tokens_fn=lambda i: 40 if (i % 2 == 0) else 200,
        )
        out["name"] = f"N2_{tag}"
        out["admission_off"] = admit_off
        out["slo_ms"] = slo
        # Fraction of attempts that completed under SLO (venue goodput proxy)
        under = float(out["admission"].get("completed_under_slo") or 0)
        out["goodput_fraction_of_attempts"] = under / max(1, n_req)
        results.append(out)
        log(
            f"  {tag}: ok={out['n_ok']} rej={out['n_rejected']} "
            f"under_slo={under:.0f} frac={out['goodput_fraction_of_attempts']:.2f} "
            f"p99={out['p99_ms']}"
        )
    return results


def exp_N3_tiers(n_req: int) -> List[Dict[str, Any]]:
    log("=== N3 Multi-tier joint cost ===")
    results = []
    for disable_tier, tag in [(False, "tier_on"), (True, "tier_off")]:
        s = Scheduler(
            strategy="nlms",
            dual=True,
            admission_off=True,
            slo_ms=1e9,
            ablation=AblationFlags(name=tag, disable_tier=disable_tier),
        )
        backends = {
            "small0": SimBackend("small0", slope=6.0, bias=40.0, tier="small"),
            "large0": SimBackend("large0", slope=20.0, bias=100.0, tier="large"),
        }
        out = simulate_load(
            s,
            backends,
            n_requests=n_req,
            tokens_fn=lambda i: 30 + (i % 10),
            tier_fn=lambda i: "large" if i % 3 == 0 else "small",
        )
        out["name"] = f"N3_{tag}"
        out["ablation"] = tag
        results.append(out)
        log(f"  {tag}: routes={out['routes']} p99={out['p99_ms']:.1f}")
    return results


def exp_N4_ablations(n_req: int) -> List[Dict[str, Any]]:
    log("=== N4 Ablations + baselines ===")
    variants: List[Tuple[str, bool, AblationFlags]] = [
        ("nlms", True, AblationFlags(name="full")),
        ("nlms", True, AblationFlags(name="no_queue", disable_queue=True)),
        ("nlms", True, AblationFlags(name="no_vram", disable_vram_soft=True, disable_vram_hard=True)),
        ("nlms", True, AblationFlags(name="no_tier", disable_tier=True)),
        ("nlms", False, AblationFlags(name="no_dual", single_timescale=True)),
        ("rls", True, AblationFlags(name="full")),
        ("static", True, AblationFlags(name="full")),
        ("round_robin", True, AblationFlags(name="full")),
        ("least_loaded", True, AblationFlags(name="full")),
    ]
    results = []
    for strategy, dual, abl in variants:
        s = Scheduler(
            strategy=strategy,
            dual=dual and not abl.single_timescale,
            ablation=abl,
            admission_off=True,
            slo_ms=1e9,
            static_slope=15.0,
            static_intercept=60.0,
        )
        backends = {
            "fast": SimBackend("fast", slope=8.0, bias=40.0, free_vram=18000),
            "slow": SimBackend(
                "slow",
                slope=22.0,
                bias=90.0,
                free_vram=3000,  # pressure for VRAM term
                thermal_start=30,
                thermal_ramp=0.02,
                thermal_max=1.5,
            ),
        }
        out = simulate_load(
            s,
            backends,
            n_requests=n_req,
            tokens_fn=lambda i: 50 + (i % 30) * 2,
        )
        out["name"] = f"N4_{strategy}_{abl.name}"
        out["strategy"] = strategy
        out["ablation"] = abl.name
        results.append(out)
        log(f"  {strategy}/{abl.name}: p99={out['p99_ms']:.1f} mape={out['prediction'].get('mape_pct')}")
    return results


def exp_N5_scale(n_workers: int, n_req: int) -> Dict[str, Any]:
    log(f"=== N5 Control-plane scale ({n_workers} backends) ===")
    s = Scheduler(strategy="nlms", dual=True, admission_off=True, slo_ms=1e9)
    backends = {
        f"w{i}": SimBackend(f"w{i}", slope=5.0 + (i % 5), bias=30.0 + i, jitter=0.05)
        for i in range(n_workers)
    }
    out = simulate_load(
        s,
        backends,
        n_requests=n_req,
        tokens_fn=lambda i: 20 + (i % 15),
    )
    # microbench pick only
    t0 = time.perf_counter()
    m = 5000
    for i in range(m):
        w, _ = s.pick("scale", tokens=25)
        s.feedback(w, 50.0, 25)
    us = (time.perf_counter() - t0) / m * 1e6
    out["name"] = "N5_scale"
    out["n_workers"] = n_workers
    out["pick_feedback_us"] = us
    log(f"  workers={n_workers} p99={out['p99_ms']:.1f} pick+fb={us:.1f}µs")
    return out


def exp_N2b_vram_hard(n_req: int = 40) -> Dict[str, Any]:
    log("=== N2b Hard VRAM admission ===")
    s = Scheduler(strategy="nlms", dual=True, admission_off=False, slo_ms=1e9)
    backends = {
        "only": SimBackend("only", slope=5.0, bias=40.0, free_vram=1500.0, total_vram=8000.0),
    }
    # long prompts → tokens > 1000 → hard block
    out = simulate_load(
        s,
        backends,
        n_requests=n_req,
        tokens_fn=lambda i: 1200,
    )
    out["name"] = "N2b_vram_hard"
    log(f"  rejected={out['n_rejected']} ok={out['n_ok']} (expect mostly rejects)")
    return out


async def exp_G1_gateway_http(duration_s: float = 8.0) -> Dict[str, Any]:
    """Optional: full HTTP path with MockBackendServer + DIOGateway."""
    import asyncio

    import httpx
    import uvicorn

    log("=== G1 Gateway HTTP path (mock engines) ===")
    fast = MockBackendServer(port=19011, latency_mult=1.0, decode_ms_per_token=4.0, name="fast")
    slow = MockBackendServer(port=19012, latency_mult=2.5, decode_ms_per_token=12.0, name="slow")
    await fast.start()
    await slow.start()
    gw = DIOGateway(
        backends=[
            Backend(id="fast", base_url=fast.base_url),
            Backend(id="slow", base_url=slow.base_url),
        ],
        strategy="nlms",
        nlms_mode="dual",
        admission_off=True,
        slo_ms=60000,
        port=18086,
    )
    config = uvicorn.Config(gw.app, host="127.0.0.1", port=18086, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait until DIO is actually listening
    async with httpx.AsyncClient(timeout=5.0) as client:
        for _ in range(40):
            try:
                h = await client.get("http://127.0.0.1:18086/healthz")
                if h.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.15)
        else:
            log("  WARNING: DIO healthz not ready")

    lats: List[float] = []
    ok = 0
    err_sample = ""
    backends_hit: Dict[str, int] = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        t_end = time.time() + duration_s
        i = 0
        while time.time() < t_end:
            t0 = time.perf_counter()
            r = await client.post(
                "http://127.0.0.1:18086/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": f"hi {i} " + ("word " * (5 + i % 20))}],
                    "max_tokens": 16,
                },
            )
            lats.append((time.perf_counter() - t0) * 1000)
            if r.status_code == 200:
                ok += 1
                b = r.headers.get("X-DIO-Backend", "?")
                backends_hit[b] = backends_hit.get(b, 0) + 1
            elif not err_sample:
                err_sample = f"{r.status_code}:{r.text[:160]}"
            i += 1

    server.should_exit = True
    await task
    await fast.stop()
    await slow.stop()
    m = gw.scheduler.metrics()
    out = {
        "name": "G1_gateway_http",
        "n_ok": ok,
        "p50_ms": pct(lats, 50),
        "p99_ms": pct(lats, 99),
        "routes": backends_hit,
        "mape_pct": m["prediction"]["mape_pct"],
        "admission": m["admission"],
        "error_sample": err_sample,
    }
    log(f"  ok={ok} p99={out['p99_ms']:.0f} routes={backends_hit} mape={out['mape_pct']:.1f} err={err_sample!r}")
    return out


def try_plots(summary: Dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed — skip figures (pip install matplotlib)")
        return

    FIGS.mkdir(parents=True, exist_ok=True)

    n1 = summary.get("N1") or []
    if n1:
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
        labels = [r["nlms_mode"] for r in n1]
        ax[0].bar(labels, [r["prediction"]["mape_pct"] or 0 for r in n1], color=["#2ca02c", "#d62728"])
        ax[0].set_title("N1 MAPE % (lower better)")
        ax[1].bar(labels, [r["p99_ms"] or 0 for r in n1], color=["#2ca02c", "#d62728"])
        ax[1].set_title("N1 p99 latency")
        fig.tight_layout()
        fig.savefig(FIGS / "N1_dual_vs_single.png", dpi=140)
        plt.close(fig)

    n2 = summary.get("N2") or []
    if n2:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        labels = ["off" if r.get("admission_off") else "on" for r in n2]
        ax.bar(labels, [r["goodput_under_slo_rps"] or 0 for r in n2], color=["#d62728", "#2ca02c"])
        ax.set_title("N2 Goodput under SLO (rps)")
        fig.tight_layout()
        fig.savefig(FIGS / "N2_admission.png", dpi=140)
        plt.close(fig)

    n4 = summary.get("N4") or []
    if n4:
        fig, ax = plt.subplots(figsize=(max(8, len(n4) * 0.55), 3.5))
        labels = [r["name"].replace("N4_", "") for r in n4]
        ax.bar(range(len(labels)), [r["p99_ms"] or 0 for r in n4], color="#1f77b4")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title("N4 Ablations / baselines p99")
        fig.tight_layout()
        fig.savefig(FIGS / "N4_ablations.png", dpi=140)
        plt.close(fig)

    log(f"Figures → {FIGS}")


def main() -> int:
    ap = argparse.ArgumentParser(description="DIO publishable novelty suite (library-first)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-gateway", action="store_true", help="Skip HTTP mock gateway test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    n = 120 if args.quick else 400
    n_scale_w = 16 if args.quick else 48
    n_scale_r = 200 if args.quick else 600

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dio_version": __import__("dio").__version__,
        "seed": args.seed,
        "mode": "library_simulation",
        "claims": [
            "dual-timescale NLMS vs single-µ (MAPE / p99 under burst+thermal)",
            "admission improves goodput under overload",
            "joint tier+vram+latency cost ablations",
            "sub-millisecond pick+feedback overhead",
            "scales to many backends with O(|W|) scan",
        ],
    }

    try:
        summary["L0"] = exp_L0_library_overhead(5000 if args.quick else 20000)
        summary["N1"] = exp_N1_dual_vs_single(n)
        summary["N2"] = exp_N2_admission(n)
        summary["N2b"] = exp_N2b_vram_hard(40 if args.quick else 80)
        summary["N3"] = exp_N3_tiers(n)
        summary["N4"] = exp_N4_ablations(max(80, n // 2))
        summary["N5"] = exp_N5_scale(n_scale_w, n_scale_r)

        if not args.skip_gateway:
            import asyncio

            summary["G1"] = asyncio.run(exp_G1_gateway_http(5.0 if args.quick else 10.0))
    except Exception as e:
        summary["error"] = str(e)
        log(f"FATAL: {e}")
        import traceback

        traceback.print_exc()

    # Flatten tables
    rows = []
    for key in ("L0", "N1", "N2", "N2b", "N3", "N4", "N5", "G1"):
        block = summary.get(key)
        if block is None:
            continue
        if isinstance(block, list):
            for r in block:
                rows.append({"suite": key, **{k: r.get(k) for k in r if k != "workers"}})
        elif isinstance(block, dict):
            rows.append({"suite": key, **{k: block.get(k) for k in block if k != "workers"}})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "tables.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    try_plots(summary)

    # Console digest
    print("\n" + "=" * 64)
    print("PUBLISHABLE SUITE DIGEST  (dio library)")
    print("=" * 64)
    if "L0" in summary:
        print(f"L0 overhead: {summary['L0']['pick_feedback_us']:.1f} µs/req")
    for r in summary.get("N1") or []:
        print(f"N1 {r['nlms_mode']}: p99={r['p99_ms']:.1f} mape={r['prediction']['mape_pct']:.2f}%")
    for r in summary.get("N2") or []:
        print(f"N2 admit_off={r['admission_off']}: ok={r['n_ok']} rej={r['n_rejected']} "
              f"goodput_slo={r['goodput_under_slo_rps']:.2f}")
    if "N5" in summary:
        print(f"N5 scale: workers={summary['N5']['n_workers']} pick={summary['N5']['pick_feedback_us']:.1f}µs")
    if "G1" in summary:
        print(f"G1 gateway: ok={summary['G1']['n_ok']} p99={summary['G1']['p99_ms']}")
    print(f"\nWrote {out_dir / 'summary.json'}")
    print("Done.")
    return 0 if "error" not in summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
