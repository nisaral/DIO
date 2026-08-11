#!/usr/bin/env python3
"""
Kaggle dual-T4 follow-ups for paper gaps (NO mock engines, NO paper mock tables).

Modes
-----
  ewma         — NLMS vs EWMA vs RR vs RLS on matched dual-T4 + delay-proxy ×2 skew
  concurrency  — concurrent in-flight 1/4/8 on dual-T4 (NLMS / RR / LL / EWMA)

Reuses workshop helpers (vLLM start, DIO serve, delay proxy, multi-seed matrix).

Example (Kaggle):
  cd /kaggle/working/DIO/dio-serve
  python scripts/kaggle/run_kaggle_followups.py --mode ewma --gpus 0,1 \\
      --model Qwen/Qwen2.5-3B-Instruct --tokenizer Qwen/Qwen2.5-3B-Instruct \\
      --out /kaggle/working/results_kaggle_ewma
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx

ROOT = Path(__file__).resolve().parents[2]  # dio-serve/
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

# Import workshop suite helpers (same dual-T4 engine path as paper)
from run_workshop_final_suite import (  # type: ignore
    Session,
    ensure_delay_proxy,
    kill_proc,
    mean_std,
    multi_seed_matrix,
    start_dio,
    start_vllm,
    wait_url,
)

PY = sys.executable


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def start_dual_vllm(
    session: Session,
    *,
    model: str,
    gpus: List[int],
    max_model_len: int = 2048,
    gpu_mem_util: float = 0.85,
) -> List[str]:
    if len(gpus) < 2:
        raise SystemExit("Need two GPU indices, e.g. --gpus 0,1")
    ports = [8000, 8001]
    urls = []
    for i, gpu in enumerate(gpus[:2]):
        port = ports[i]
        start_vllm(
            session,
            gpu=str(gpu),
            port=port,
            model=model,
            max_model_len=max_model_len,
            gpu_mem_util=gpu_mem_util,
            name=f"vllm_gpu{gpu}",
            enable_prefix_caching=False,
        )
        url = f"http://127.0.0.1:{port}"
        if not wait_url(url + "/v1/models", timeout=900):
            raise RuntimeError(f"vLLM on GPU {gpu} port {port} failed to start")
        urls.append(url)
        log(f"vLLM up gpu={gpu} {url}")
    return urls


def write_snippets_ewma(summary: Dict[str, Any], outdir: Path) -> None:
    lines = [
        "# Kaggle EWMA baseline (dual-T4 real vLLM)",
        f"generated: {summary.get('generated_at')}",
        "",
    ]
    for cell_name in ("matched", "skew_delay_x2"):
        cell = summary.get(cell_name) or {}
        lines.append(f"## {cell_name}")
        for strat, block in (cell.get("strategies") or {}).items():
            p99 = block.get("e2e_p99") or {}
            imp = block.get("p99_improvement_vs_rr_pct") or {}
            frac = block.get("frac_e0") or {}
            lines.append(
                f"- {strat}: p99={p99.get('mean')}±{p99.get('std')} "
                f"frac_e0={frac.get('mean')}±{frac.get('std')} "
                f"vsRR={imp.get('mean')}±{imp.get('std')}"
            )
        lines.append("")
    (outdir / "paper_snippets.md").write_text("\n".join(lines), encoding="utf-8")


def run_ewma_mode(args: argparse.Namespace) -> Dict[str, Any]:
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    session = Session(outdir)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    seeds = 2 if args.quick else args.seeds
    n_per = 10 if args.quick else args.n_per_seed
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    summary: Dict[str, Any] = {
        "mode": "ewma",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "gpus": gpus,
        "seeds": seeds,
        "n_per_seed": n_per,
        "strategies": strategies,
        "note": "Real dual-GPU vLLM; EWMA is e2e-latency EMA ranker (no token feature).",
    }

    try:
        backends = start_dual_vllm(session, model=args.model, gpus=gpus)

        # Matched dual-T4 (homogeneous)
        summary["matched"] = multi_seed_matrix(
            session,
            backends,
            strategies=strategies,
            seeds=seeds,
            n_per_seed=n_per,
            max_tokens=args.max_tokens,
            model=args.model,
            tokenizer=args.tokenizer or "",
            dio_base=9100,
            label="kaggle_ewma_matched",
        )

        # Skew: delay-proxy ×slow_mult on backend e1
        ensure_delay_proxy()
        proxy_port = 8101
        raw_e1 = backends[1]
        proxy_cmd = [
            PY,
            str(ROOT / "scripts" / "latency_delay_proxy.py"),
            "--upstream",
            raw_e1,
            "--port",
            str(proxy_port),
            "--mult",
            str(args.slow_mult),
        ]
        session.start("delay_proxy", proxy_cmd)
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        if not wait_url(proxy_url + "/v1/models", timeout=60):
            # some proxies only forward /v1/chat; try health via chat skip
            log("WARNING: proxy /v1/models slow; continuing")
        skewed = [backends[0], proxy_url]
        summary["skew_delay_x2"] = multi_seed_matrix(
            session,
            skewed,
            strategies=strategies,
            seeds=seeds,
            n_per_seed=n_per,
            max_tokens=args.max_tokens,
            model=args.model,
            tokenizer=args.tokenizer or "",
            dio_base=9200,
            label="kaggle_ewma_skew",
        )
        summary["skew_note"] = f"e1 behind delay-proxy ×{args.slow_mult} (not multi-SKU)"
    finally:
        session.cleanup()

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_snippets_ewma(summary, outdir)
    log(f"wrote {outdir / 'summary.json'}")
    return summary


def concurrent_load(
    base: str,
    *,
    model: str,
    n: int,
    max_tokens: int,
    seed: int,
    concurrent: int,
) -> Dict[str, Any]:
    """Fire n chat completions with up to `concurrent` in flight."""
    prompts = [
        "What is 2+2? One sentence.",
        "Name three colors briefly.",
        "Explain gravity in one sentence.",
        "List two benefits of exercise.",
        "What is the capital of France?",
    ]
    t0_wall = time.perf_counter()
    limits = httpx.Limits(
        max_connections=max(16, concurrent * 4),
        max_keepalive_connections=max(16, concurrent * 4),
    )

    def one(i: int) -> Dict[str, Any]:
        prompt = prompts[i % len(prompts)] + f" (seed={seed} i={i})"
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=300.0, limits=limits) as client:
                r = client.post(
                    f"{base.rstrip('/')}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                    },
                )
                ms = (time.perf_counter() - t0) * 1000.0
                if r.status_code == 200:
                    return {
                        "ok": True,
                        "ms": ms,
                        "wid": r.headers.get("X-DIO-Backend") or "unknown",
                    }
                return {"ok": False, "ms": ms, "wid": None}
        except Exception:
            return {"ok": False, "ms": (time.perf_counter() - t0) * 1000.0, "wid": None}

    if concurrent <= 1:
        results = [one(i) for i in range(n)]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=concurrent) as ex:
            futs = [ex.submit(one, i) for i in range(n)]
            for f in as_completed(futs):
                results.append(f.result())

    e2e = [r["ms"] for r in results if r.get("ok")]
    routes: Dict[str, int] = {}
    for r in results:
        if r.get("ok") and r.get("wid"):
            routes[r["wid"]] = routes.get(r["wid"], 0) + 1
    ok = sum(1 for r in results if r.get("ok"))
    fail = len(results) - ok
    wall = max(1e-6, time.perf_counter() - t0_wall)
    total = sum(routes.values()) or 1
    return {
        "seed": seed,
        "concurrent": concurrent,
        "ok": ok,
        "fail": fail,
        "e2e_p50_ms": pct(e2e, 50),
        "e2e_p99_ms": pct(e2e, 99),
        "e2e_mean_ms": statistics.mean(e2e) if e2e else None,
        "frac_e0": routes.get("e0", 0) / total,
        "routes": routes,
        "throughput_rps": ok / wall,
        "wall_s": wall,
    }


def run_concurrency_mode(args: argparse.Namespace) -> Dict[str, Any]:
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    session = Session(outdir)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    seeds = 2 if args.quick else args.seeds
    n_per = 12 if args.quick else args.n_per_seed
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    concs = [int(x) for x in args.concurrencies.split(",") if x.strip()]

    summary: Dict[str, Any] = {
        "mode": "concurrency",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "gpus": gpus,
        "seeds": seeds,
        "n_per_seed": n_per,
        "concurrencies": concs,
        "strategies": strategies,
        "note": "Real dual-GPU concurrent load; not multi-SKU Regime D.",
        "cells": {},
    }

    try:
        backends = start_dual_vllm(session, model=args.model, gpus=gpus)
        dio_port = 9300
        for conc in concs:
            for strat in strategies:
                key = f"c{conc}_{strat}"
                log(f"=== concurrent={conc} strategy={strat} ===")
                rows = []
                for seed in range(seeds):
                    port = dio_port
                    dio_port += 1
                    url = start_dio(
                        session,
                        backends=backends,
                        port=port,
                        strategy=strat,
                        name=f"conc_{conc}_{strat}_s{seed}",
                        tokenizer=args.tokenizer or "",
                        admission_mode="rank_only",
                        admission_off=True,
                    )
                    if not wait_url(url + "/healthz", timeout=90):
                        rows.append({"seed": seed, "error": "dio_start_failed"})
                        if session.handles:
                            kill_proc(session.handles.pop())
                        continue
                    row = concurrent_load(
                        url,
                        model=args.model,
                        n=n_per,
                        max_tokens=args.max_tokens,
                        seed=seed,
                        concurrent=conc,
                    )
                    row["strategy"] = strat
                    rows.append(row)
                    log(
                        f"  seed={seed} p99={row.get('e2e_p99_ms')} "
                        f"frac_e0={row.get('frac_e0')} rps={row.get('throughput_rps')}"
                    )
                    if session.handles:
                        kill_proc(session.handles.pop())
                    time.sleep(0.4)

                p99s = [r["e2e_p99_ms"] for r in rows if r.get("e2e_p99_ms") is not None]
                p50s = [r["e2e_p50_ms"] for r in rows if r.get("e2e_p50_ms") is not None]
                fracs = [r["frac_e0"] for r in rows if r.get("frac_e0") is not None]
                rps = [r["throughput_rps"] for r in rows if r.get("throughput_rps") is not None]
                summary["cells"][key] = {
                    "concurrent": conc,
                    "strategy": strat,
                    "p99": mean_std(p99s),
                    "p50": mean_std(p50s),
                    "frac_e0": mean_std(fracs),
                    "throughput_rps": mean_std(rps),
                    "per_seed": rows,
                }
    finally:
        session.cleanup()

    # vs RR
    vs: Dict[str, Any] = {}
    for conc in concs:
        rr = summary["cells"].get(f"c{conc}_round_robin")
        if not rr:
            continue
        rr_rows = rr.get("per_seed") or []
        for strat in strategies:
            if strat == "round_robin":
                continue
            cell = summary["cells"].get(f"c{conc}_{strat}")
            if not cell:
                continue
            imps = []
            wins = 0
            for a, b in zip(cell.get("per_seed") or [], rr_rows):
                if a.get("e2e_p99_ms") and b.get("e2e_p99_ms") and b["e2e_p99_ms"] > 0:
                    imp = 100.0 * (b["e2e_p99_ms"] - a["e2e_p99_ms"]) / b["e2e_p99_ms"]
                    imps.append(imp)
                    if imp > 0:
                        wins += 1
            vs[f"c{conc}_{strat}"] = {
                **mean_std(imps),
                "wins": wins,
                "of": len(imps),
            }
    summary["p99_improvement_vs_rr"] = vs

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Kaggle concurrency (dual-T4 real vLLM)", ""]
    for conc in concs:
        lines.append(f"## concurrent={conc}")
        for strat in strategies:
            c = summary["cells"].get(f"c{conc}_{strat}")
            if not c:
                continue
            imp = vs.get(f"c{conc}_{strat}")
            extra = ""
            if imp and imp.get("mean") is not None:
                extra = f" vsRR={imp['mean']:.1f}±{imp.get('std', 0):.1f} ({imp['wins']}/{imp['of']})"
            lines.append(
                f"- {strat}: p99={c['p99'].get('mean')}±{c['p99'].get('std')} "
                f"frac_e0={c['frac_e0'].get('mean')} rps={c['throughput_rps'].get('mean')}{extra}"
            )
        lines.append("")
    (outdir / "paper_snippets.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {outdir / 'summary.json'}")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Kaggle dual-T4 follow-ups (EWMA / concurrency)")
    p.add_argument("--mode", choices=("ewma", "concurrency", "both"), default="ewma")
    p.add_argument("--engine-mode", default="vllm", help="kept for CLI compatibility")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-per-seed", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--slow-mult", type=float, default=2.0, help="delay-proxy multiplier for ewma skew cell")
    p.add_argument(
        "--strategies",
        default="nlms,ewma,round_robin,rls",
        help="comma list (ewma mode default includes ewma)",
    )
    p.add_argument("--concurrencies", default="1,4,8")
    p.add_argument("--out", default=str(ROOT / "results_kaggle_followups"))
    p.add_argument("--quick", action="store_true", help="2 seeds / fewer requests")
    args = p.parse_args()

    if args.mode in ("ewma", "both"):
        if "ewma" not in args.strategies:
            args.strategies = "nlms,ewma,round_robin,rls"
        out_e = args.out if args.mode == "ewma" else str(Path(args.out) / "ewma")
        args_e = argparse.Namespace(**{**vars(args), "out": out_e})
        run_ewma_mode(args_e)

    if args.mode in ("concurrency", "both"):
        if args.mode == "concurrency" and args.strategies == "nlms,ewma,round_robin,rls":
            args.strategies = "nlms,ewma,round_robin,least_loaded"
        out_c = args.out if args.mode == "concurrency" else str(Path(args.out) / "concurrency")
        args_c = argparse.Namespace(**{**vars(args), "out": out_c})
        run_concurrency_mode(args_c)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
