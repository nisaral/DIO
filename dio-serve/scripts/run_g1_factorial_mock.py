#!/usr/bin/env python3
"""
G1 2x2 factorial on mock engines: separates the engine-metrics effect from the
session-affinity effect that the published 2-arm design conflated.

This is a MECHANISM study, not a replacement for the dual-T4 numbers. Mock
engines give what the GPU campaign could not: a controlled setting where the
scraped gauges are provably non-zero (real queueing, real prefix-cache
accounting), so each knob's contribution is separately observable.

Arms:
    hybrid_on      metrics ON   affinity ON
    metrics_only   metrics ON   affinity OFF
    affinity_only  metrics OFF  affinity ON
    hybrid_off     metrics OFF  affinity OFF
    rr             round-robin baseline

Usage:
    python scripts/run_g1_factorial_mock.py --seeds 5 --concurrent 8
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_gpu_abc_suite import run_multiturn_load  # type: ignore

PY = sys.executable
BACKENDS = [("e0", 8000, 120.0), ("e1", 8001, 240.0)]
GW_PORT = 9100

ARMS = [
    {"name": "hybrid_on", "strategy": "nlms", "metrics": True, "bonus": 200.0},
    {"name": "metrics_only", "strategy": "nlms", "metrics": True, "bonus": 0.0},
    {"name": "affinity_only", "strategy": "nlms", "metrics": False, "bonus": 200.0},
    {"name": "hybrid_off", "strategy": "nlms", "metrics": False, "bonus": 0.0},
    {"name": "rr", "strategy": "round_robin", "metrics": False, "bonus": 0.0},
]


def log(msg: str) -> None:
    print(f"[factorial] {msg}", flush=True)


def wait_url(url: str, timeout: float = 60.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def reset_backends() -> None:
    for _, port, _ in BACKENDS:
        try:
            httpx.post(f"http://127.0.0.1:{port}/debug/reset", timeout=5.0)
        except Exception:
            pass


def backend_peaks() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, port, _ in BACKENDS:
        try:
            out[name] = httpx.get(
                f"http://127.0.0.1:{port}/debug/peaks", timeout=5.0
            ).json()
        except Exception:
            out[name] = {}
    return out


def start_mocks() -> List[subprocess.Popen]:
    procs = []
    for _, port, lat in BACKENDS:
        p = subprocess.Popen(
            [
                PY, str(ROOT / "scripts" / "mock_vllm_server.py"),
                "--port", str(port),
                "--latency-ms", str(lat),
                "--max-concurrency", "4",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)
    for _, port, _ in BACKENDS:
        if not wait_url(f"http://127.0.0.1:{port}/v1/models", 90):
            raise RuntimeError(f"mock backend {port} not ready")
    return procs


def start_gateway(arm: Dict[str, Any]) -> subprocess.Popen:
    env = {
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONIOENCODING": "utf-8",
        "DIO_ENGINE_METRICS": "1" if arm["metrics"] else "0",
        "DIO_CACHE_BONUS_MS": str(arm["bonus"]),
        "DIO_METRICS_INTERVAL_S": "0.1",
        "DIO_ADMISSION_OFF": "1",
    }
    import os

    full_env = dict(os.environ)
    full_env.update(env)
    cmd = [
        PY, "-m", "dio", "serve",
        "--host", "127.0.0.1", "--port", str(GW_PORT),
        "--strategy", arm["strategy"], "--nlms-mode", "dual",
        "--slo-ms", "180000", "--admission-mode", "rank_only", "--admission-off",
    ]
    for name, port, _ in BACKENDS:
        cmd.extend(["-b", f"{name}=http://127.0.0.1:{port}"])
    p = subprocess.Popen(
        cmd, env=full_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if not wait_url(f"http://127.0.0.1:{GW_PORT}/debug/engine", 90):
        raise RuntimeError("gateway not ready")
    return p


def mean_std(xs: List[float]) -> Dict[str, Any]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.mean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "n": len(xs),
        "median": statistics.median(xs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--concurrent", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results_g1_factorial_mock"))
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    mocks = start_mocks()
    log("mock backends up")
    results: Dict[str, Any] = {"arms": {}, "config": vars(args)}

    try:
        for arm in ARMS:
            per_seed = []
            gw = start_gateway(arm)
            log(f"arm={arm['name']} metrics={arm['metrics']} bonus={arm['bonus']}")
            try:
                for seed in range(args.seeds):
                    reset_backends()
                    r = run_multiturn_load(
                        f"http://127.0.0.1:{GW_PORT}",
                        model="mock-model",
                        n_sessions=args.sessions,
                        turns=args.turns,
                        max_tokens=args.max_tokens,
                        seed=seed,
                        concurrent=args.concurrent,
                    )
                    r["backend_peaks"] = backend_peaks()
                    per_seed.append(r)
                    log(
                        f"  seed={seed} p99={r['e2e_p99_ms']:.0f}ms "
                        f"frac_e0={r['frac_e0']:.2f} stick={r['stickiness']:.2f} "
                        f"peakrun={max((v.get('peak_running',0) or 0) for v in r['backend_peaks'].values())}"
                    )
            finally:
                gw.terminate()
                try:
                    gw.wait(timeout=15)
                except Exception:
                    gw.kill()
                time.sleep(1.0)

            results["arms"][arm["name"]] = {
                "config": arm,
                "per_seed": per_seed,
                "p99": mean_std([r["e2e_p99_ms"] for r in per_seed]),
                "p50": mean_std([r["e2e_p50_ms"] for r in per_seed]),
                "frac_e0": mean_std([r["frac_e0"] for r in per_seed]),
                "stickiness": mean_std([r["stickiness"] for r in per_seed]),
            }
    finally:
        for p in mocks:
            p.terminate()

    # paired margins
    def paired(base: str, treat: str) -> Dict[str, Any]:
        b = results["arms"].get(base, {}).get("per_seed", [])
        t = results["arms"].get(treat, {}).get("per_seed", [])
        gains = []
        for i in range(min(len(b), len(t))):
            x, y = b[i]["e2e_p99_ms"], t[i]["e2e_p99_ms"]
            if x and y and x > 0:
                gains.append(100.0 * (x - y) / x)
        return mean_std(gains)

    results["margins"] = {
        "engine_metrics_effect_pct": paired("affinity_only", "hybrid_on"),
        "affinity_effect_pct": paired("metrics_only", "hybrid_on"),
        "metrics_only_vs_off_pct": paired("hybrid_off", "metrics_only"),
        "affinity_only_vs_off_pct": paired("hybrid_off", "affinity_only"),
        "hybrid_on_vs_rr_pct": paired("rr", "hybrid_on"),
    }

    (outdir / "summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 74)
    print("G1 2x2 FACTORIAL (mock engines, provably non-zero gauges)")
    print("=" * 74)
    hdr = f"{'arm':<15}{'p99 ms':>16}{'frac_e0':>10}{'stick':>8}{'peak_run':>10}"
    print(hdr)
    for name, blk in results["arms"].items():
        pk = max(
            (v.get("peak_running", 0) or 0)
            for r in blk["per_seed"]
            for v in r["backend_peaks"].values()
        )
        print(
            f"{name:<15}{blk['p99']['mean']:>10.0f}±{blk['p99']['std']:<5.0f}"
            f"{blk['frac_e0']['mean']:>10.2f}{blk['stickiness']['mean']:>8.2f}{pk:>10.0f}"
        )
    print("\nPaired margins (positive = treatment better):")
    for k, v in results["margins"].items():
        if v["mean"] is not None:
            print(f"  {k:<28} {v['mean']:+7.2f}% ± {v['std']:.2f} (n={v['n']})")
    print(f"\nwrote {outdir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
