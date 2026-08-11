#!/usr/bin/env python3
"""
LOCAL-ONLY mock smoke (DO NOT cite in the paper).

For paper-facing dual-T4 EWMA + concurrency, use Kaggle instead:
  scripts/kaggle/KAGGLE_RUNBOOK.md
  scripts/kaggle/run_kaggle_followups.py

This script only exercises the gateway against mock_vllm_server for laptop
debugging. Reviewers should never see mock numbers as Regime D evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PY = sys.executable
# Slow peer ≈ T4-ish, fast peer ≈ A30-ish (absolute ms are mock, not real decode).
SLOW_PORT, FAST_PORT = 18201, 18202
SLOW_LAT_MS, FAST_LAT_MS = 240.0, 90.0
GW_PORT = 18280
MODEL = "mock-model"
STRATEGIES = ("nlms", "rls", "ewma", "round_robin", "least_loaded")
CONCURRENCIES = (1, 4, 8)
MAX_TOKENS_MIX = (32, 64, 128, 256)


def log(msg: str) -> None:
    print(f"[mock-hetero] {msg}", flush=True)


def wait_url(url: str, timeout: float = 60.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def start_mock(port: int, latency_ms: float, max_conc: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            PY,
            str(ROOT / "scripts" / "mock_vllm_server.py"),
            "--port",
            str(port),
            "--latency-ms",
            str(latency_ms),
            "--max-concurrency",
            str(max_conc),
            "--model",
            MODEL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_gateway(strategy: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONIOENCODING": "utf-8",
            "DIO_ENGINE_METRICS": "0",
            "DIO_ADMISSION_OFF": "1",
            "DIO_CACHE_BONUS_MS": "0",
        }
    )
    cmd = [
        PY,
        "-m",
        "dio",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(GW_PORT),
        "--strategy",
        strategy,
        "--nlms-mode",
        "dual",
        "--slo-ms",
        "180000",
        "--admission-mode",
        "rank_only",
        "--admission-off",
        "-b",
        f"slow=http://127.0.0.1:{SLOW_PORT}",
        "-b",
        f"fast=http://127.0.0.1:{FAST_PORT}",
        "--vram",
        "16000",
        "--vram",
        "24000",
    ]
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_url(f"http://127.0.0.1:{GW_PORT}/v1/models", 90):
        p.kill()
        raise RuntimeError(f"gateway not ready for strategy={strategy}")
    return p


def stop(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (p / 100.0) * (len(ys) - 1)
    lo = int(k)
    hi = min(len(ys) - 1, lo + 1)
    t = k - lo
    return ys[lo] * (1.0 - t) + ys[hi] * t


def one_request(
    client: httpx.Client,
    *,
    seed: int,
    idx: int,
    max_tokens: int,
) -> Tuple[bool, float, Optional[str]]:
    prompt = (
        f"seed={seed} idx={idx} "
        + ("padding " * (8 + (idx % 12)))
        + "Summarize the trade-offs of multi-instance LLM routing."
    )
    t0 = time.perf_counter()
    try:
        r = client.post(
            f"http://127.0.0.1:{GW_PORT}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=120.0,
        )
        e2e = (time.perf_counter() - t0) * 1000.0
        if r.status_code >= 400:
            return False, e2e, None
        # Prefer gateway decision log if present; else infer from latency heuristics.
        worker = None
        try:
            dbg = client.get(f"http://127.0.0.1:{GW_PORT}/debug/metrics", timeout=5.0).json()
            decs = dbg.get("decisions") or []
            if decs:
                worker = (decs[-1] or {}).get("worker_id")
        except Exception:
            pass
        return True, e2e, worker
    except Exception:
        return False, (time.perf_counter() - t0) * 1000.0, None


def run_cell(
    *,
    strategy: str,
    concurrent: int,
    seed: int,
    n_requests: int,
) -> Dict[str, Any]:
    # Reset gateway stats between cells
    try:
        httpx.post(f"http://127.0.0.1:{GW_PORT}/debug/reset_stats", timeout=5.0)
    except Exception:
        pass
    for port in (SLOW_PORT, FAST_PORT):
        try:
            httpx.post(f"http://127.0.0.1:{port}/debug/reset", timeout=5.0)
        except Exception:
            pass

    e2e: List[float] = []
    routes: Dict[str, int] = {"slow": 0, "fast": 0, "unknown": 0}
    ok = 0
    fail = 0
    t_wall0 = time.perf_counter()

    limits = httpx.Limits(
        max_connections=max(16, concurrent * 4),
        max_keepalive_connections=max(16, concurrent * 4),
    )
    with httpx.Client(timeout=120.0, limits=limits) as client:
        jobs = []
        for i in range(n_requests):
            mt = MAX_TOKENS_MIX[i % len(MAX_TOKENS_MIX)]
            jobs.append((i, mt))

        def work(item: Tuple[int, int]) -> Tuple[bool, float, Optional[str]]:
            i, mt = item
            return one_request(client, seed=seed, idx=i, max_tokens=mt)

        if concurrent <= 1:
            results = [work(j) for j in jobs]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=concurrent) as ex:
                futs = [ex.submit(work, j) for j in jobs]
                for f in as_completed(futs):
                    results.append(f.result())

        # Prefer full decision log for accurate route fractions
        try:
            dbg = client.get(f"http://127.0.0.1:{GW_PORT}/debug/metrics", timeout=5.0).json()
            decs = dbg.get("decisions") or []
            if len(decs) >= n_requests // 2:
                routes = {"slow": 0, "fast": 0, "unknown": 0}
                for d in decs[-n_requests:]:
                    wid = (d or {}).get("worker_id") or "unknown"
                    if wid not in routes:
                        routes[wid] = 0
                    routes[wid] = routes.get(wid, 0) + 1
        except Exception:
            pass

        for success, lat, worker in results:
            if success:
                ok += 1
                e2e.append(lat)
                if worker in ("slow", "fast"):
                    # only if log path didn't already fill routes completely
                    pass
            else:
                fail += 1

    wall_s = max(1e-6, time.perf_counter() - t_wall0)
    total_r = sum(routes.values()) or 1
    # If decisions empty, fall back to ok count as unknown
    if total_r == 1 and routes.get("unknown", 0) == 0 and sum(routes.values()) == 0:
        routes = {"slow": 0, "fast": 0, "unknown": ok}

    return {
        "strategy": strategy,
        "concurrent": concurrent,
        "seed": seed,
        "ok": ok,
        "fail": fail,
        "e2e_p50_ms": pct(e2e, 50),
        "e2e_p95_ms": pct(e2e, 95),
        "e2e_p99_ms": pct(e2e, 99),
        "e2e_mean_ms": statistics.mean(e2e) if e2e else float("nan"),
        "routes": routes,
        "frac_fast": routes.get("fast", 0) / total_r,
        "frac_slow": routes.get("slow", 0) / total_r,
        "throughput_rps": ok / wall_s,
        "wall_s": wall_s,
    }


def mean_std(xs: List[float]) -> Dict[str, Any]:
    xs = [float(x) for x in xs if x is not None and x == x]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.mean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "n": len(xs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument(
        "--concurrencies",
        default="1,4,8",
        help="comma list of concurrent in-flight request counts",
    )
    ap.add_argument(
        "--strategies",
        default="nlms,rls,ewma,round_robin,least_loaded",
    )
    ap.add_argument("--out", default=str(ROOT / "results_mock_hetero"))
    ap.add_argument("--mock-max-concurrency", type=int, default=16)
    args = ap.parse_args()

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    mocks = [
        start_mock(SLOW_PORT, SLOW_LAT_MS, args.mock_max_concurrency),
        start_mock(FAST_PORT, FAST_LAT_MS, args.mock_max_concurrency),
    ]
    for port in (SLOW_PORT, FAST_PORT):
        if not wait_url(f"http://127.0.0.1:{port}/v1/models", 60):
            for p in mocks:
                stop(p)
            raise RuntimeError(f"mock {port} not ready")
    log(f"mocks up slow={SLOW_LAT_MS}ms fast={FAST_LAT_MS}ms")

    summary: Dict[str, Any] = {
        "note": "Mock hetero mechanism study; not physical Regime D.",
        "slow_latency_ms": SLOW_LAT_MS,
        "fast_latency_ms": FAST_LAT_MS,
        "requests_per_seed": args.requests,
        "seeds": args.seeds,
        "cells": {},
    }

    try:
        for conc in concurrencies:
            for strat in strategies:
                key = f"c{conc}_{strat}"
                log(f"=== concurrent={conc} strategy={strat} ===")
                gw = start_gateway(strat)
                per_seed: List[Dict[str, Any]] = []
                try:
                    for seed in range(args.seeds):
                        row = run_cell(
                            strategy=strat,
                            concurrent=conc,
                            seed=seed,
                            n_requests=args.requests,
                        )
                        per_seed.append(row)
                        log(
                            f"  seed={seed} p99={row['e2e_p99_ms']:.0f} "
                            f"frac_fast={row['frac_fast']:.2f} "
                            f"rps={row['throughput_rps']:.1f}"
                        )
                finally:
                    stop(gw)
                    time.sleep(0.6)

                p99s = [r["e2e_p99_ms"] for r in per_seed]
                p50s = [r["e2e_p50_ms"] for r in per_seed]
                fracs = [r["frac_fast"] for r in per_seed]
                rps = [r["throughput_rps"] for r in per_seed]
                summary["cells"][key] = {
                    "concurrent": conc,
                    "strategy": strat,
                    "per_seed": per_seed,
                    "p50": mean_std(p50s),
                    "p99": mean_std(p99s),
                    "frac_fast": mean_std(fracs),
                    "throughput_rps": mean_std(rps),
                }
    finally:
        for p in mocks:
            stop(p)

    # vs RR improvements at each concurrency
    vs_rr: Dict[str, Any] = {}
    for conc in concurrencies:
        rr = summary["cells"].get(f"c{conc}_round_robin")
        if not rr:
            continue
        rr_p99 = [r["e2e_p99_ms"] for r in rr["per_seed"]]
        for strat in strategies:
            if strat == "round_robin":
                continue
            cell = summary["cells"].get(f"c{conc}_{strat}")
            if not cell:
                continue
            imps = []
            wins = 0
            for a, b in zip(cell["per_seed"], rr["per_seed"]):
                if b["e2e_p99_ms"] > 0:
                    imp = 100.0 * (b["e2e_p99_ms"] - a["e2e_p99_ms"]) / b["e2e_p99_ms"]
                    imps.append(imp)
                    if imp > 0:
                        wins += 1
            vs_rr[f"c{conc}_{strat}"] = {
                "mean": statistics.mean(imps) if imps else None,
                "std": statistics.pstdev(imps) if len(imps) > 1 else 0.0,
                "wins": wins,
                "of": len(imps),
            }
    summary["p99_improvement_vs_rr"] = vs_rr

    out_json = outdir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {out_json}")

    # Compact markdown for paper drafting
    lines = [
        "# Mock hetero concurrency + EWMA (mechanism study)",
        f"slow={SLOW_LAT_MS}ms fast={FAST_LAT_MS}ms seeds={args.seeds} reqs={args.requests}",
        "",
    ]
    for conc in concurrencies:
        lines.append(f"## concurrent={conc}")
        for strat in strategies:
            cell = summary["cells"].get(f"c{conc}_{strat}")
            if not cell:
                continue
            imp = vs_rr.get(f"c{conc}_{strat}")
            imp_s = (
                f" vsRR p99 {imp['mean']:.1f}%±{imp['std']:.1f} ({imp['wins']}/{imp['of']})"
                if imp and imp["mean"] is not None
                else ""
            )
            lines.append(
                f"- {strat}: p99={cell['p99']['mean']:.0f}±{cell['p99']['std']:.0f} "
                f"frac_fast={cell['frac_fast']['mean']:.3f}±{cell['frac_fast']['std']:.3f} "
                f"rps={cell['throughput_rps']['mean']:.1f}{imp_s}"
            )
        lines.append("")
    (outdir / "paper_snippets.md").write_text("\n".join(lines), encoding="utf-8")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
