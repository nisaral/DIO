#!/usr/bin/env python3
"""
NLMS vs RLS vs RR vs LL head-to-head (paper related-work gap).

Two modes:
  1) --mode sim   (default, no GPU): slope-skew library bench + overhead µs
  2) --mode real  : wrap existing dual engines (or start vLLM) like hetero runner

Outputs mean±std p99, frac_to_fast, MAPE, and pick() wall time (µs).

Kaggle (after engines up or with vllm):
  python scripts/run_rls_headtohead.py --mode real --gpus 0,1 --seeds 3 \\
      --tokenizer Qwen/Qwen2.5-3B-Instruct --out results_rls_headtohead
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dio.scheduler import Scheduler  # noqa: E402


def mean_std(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    # rough 95% CI half-width (normal approx)
    ci = 1.96 * s / (len(xs) ** 0.5) if len(xs) > 1 else 0.0
    return {"mean": m, "std": s, "ci95": ci, "n": len(xs)}


def bench_overhead_us(strategy: str, n: int = 5000) -> float:
    s = Scheduler(strategy=strategy, admission_off=True, slo_ms=1e9)
    s.register("e0")
    s.register("e1")
    # warm
    for _ in range(50):
        s.pick("warmup", tokens=64)
    t0 = time.perf_counter()
    for i in range(n):
        s.pick(f"p{i}", tokens=32 + (i % 40))
    return (time.perf_counter() - t0) * 1e6 / n


def sim_seed(strategy: str, seed: int, n: int = 200) -> Dict[str, Any]:
    """Slope-skew sim: fast vs slow workers (same cost function, different predictor)."""
    random.seed(seed)
    s = Scheduler(strategy=strategy, admission_off=True, slo_ms=1e9, dual=True)
    s.register("fast")
    s.register("slow")
    # true latency models
    true = {"fast": (2.0, 40.0), "slow": (8.0, 90.0)}  # slope, intercept
    e2e: List[float] = []
    routes: Dict[str, int] = {}
    for i in range(n):
        tokens = 40 + (i % 60)
        prompt = "x" * (tokens * 4)
        wid, dec = s.pick(prompt, tokens=tokens)
        slope, b = true[wid]
        # jitter
        actual = (slope * tokens + b) * random.uniform(0.92, 1.08)
        s.feedback(wid, actual, tokens)
        e2e.append(actual)
        routes[wid] = routes.get(wid, 0) + 1
    total = sum(routes.values()) or 1
    ys = sorted(e2e)
    p99 = ys[min(len(ys) - 1, int(round(0.99 * (len(ys) - 1))))]
    m = s.metrics()["prediction"]
    return {
        "seed": seed,
        "strategy": strategy,
        "frac_fast": routes.get("fast", 0) / total,
        "p99": p99,
        "mean": statistics.mean(e2e),
        "mape": m.get("mape_pct"),
        "routes": routes,
    }


def run_sim(seeds: int, n: int) -> Dict[str, Any]:
    strategies = ["nlms", "rls", "round_robin", "least_loaded"]
    out: Dict[str, Any] = {"strategies": {}, "overhead_us": {}}
    for strat in strategies:
        rows = [sim_seed(strat, seed, n) for seed in range(seeds)]
        out["strategies"][strat] = {
            "frac_fast": mean_std([r["frac_fast"] for r in rows]),
            "p99": mean_std([r["p99"] for r in rows]),
            "mape": mean_std([float(r["mape"] or 0) for r in rows]),
            "per_seed": rows,
        }
        # p99 improvement vs RR
        if strat != "round_robin":
            imps = []
            for i in range(seeds):
                rr = out["strategies"].get("round_robin")
                # compute after RR filled — second pass below
            out["strategies"][strat]["_rows"] = rows
        out["overhead_us"][strat] = bench_overhead_us(strat)

    # fill RR first for imps
    rr_p99s = [r["p99"] for r in out["strategies"]["round_robin"]["per_seed"]]
    for strat in strategies:
        if strat == "round_robin":
            continue
        imps = []
        for i, r in enumerate(out["strategies"][strat]["per_seed"]):
            if rr_p99s[i] > 0:
                imps.append((rr_p99s[i] - r["p99"]) / rr_p99s[i] * 100.0)
        out["strategies"][strat]["p99_improvement_vs_rr_pct"] = mean_std(imps)
        out["strategies"][strat].pop("_rows", None)

    return out


def run_real(args) -> Dict[str, Any]:
    """Reuse run_real_hetero machinery with multiple strategies."""
    # Import from sibling script
    import importlib.util

    path = ROOT / "scripts" / "run_real_hetero_multiseed.py"
    spec = importlib.util.spec_from_file_location("hetero", path)
    assert spec and spec.loader
    hetero = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hetero)

    session = hetero.Session(Path(args.out))
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    summary: Dict[str, Any] = {
        "mode": "real",
        "strategies": {},
        "args": vars(args),
    }
    try:
        # engines
        backends: List[str] = []
        if args.backends.strip():
            backends = [b.strip() for b in args.backends.split(",") if b.strip()]
        else:
            gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
            for i, gpu in enumerate(gpus[:2]):
                port = args.engine_base_port + i
                hetero.start_vllm(
                    session, gpu, port, args.model,
                    args.max_model_len, args.gpu_mem_util, f"vllm_gpu{gpu}",
                )
                backends.append(f"http://127.0.0.1:{port}")
            for b in backends:
                if not hetero.wait_url(b + "/v1/models", timeout=900):
                    raise RuntimeError(f"engine failed {b}")

        # optional throttle for Regime C style
        if args.slow_mult > 1.0 and len(backends) >= 2:
            hetero.ensure_delay_proxy_script()
            session.start(
                "delay_proxy",
                [
                    hetero.PY,
                    str(hetero.DELAY_PROXY),
                    "--upstream",
                    backends[1],
                    "--port",
                    str(args.proxy_port),
                    "--latency-mult",
                    str(args.slow_mult),
                ],
            )
            if not hetero.wait_url(f"http://127.0.0.1:{args.proxy_port}/health", timeout=60):
                raise RuntimeError("proxy failed")
            backends = [backends[0], f"http://127.0.0.1:{args.proxy_port}"]

        for strat in strategies:
            rows = []
            for seed in range(args.seeds):
                port = args.dio_base_port + hash(strat) % 100 + seed
                # start dio with strategy + tokenizer
                cmd = [
                    hetero.PY, "-m", "dio", "serve",
                    "--port", str(port), "--host", "127.0.0.1",
                    "--strategy", strat, "--nlms-mode", "dual",
                    "--slo-ms", "180000", "--admission-off",
                    "--admission-mode", "rank_only",
                ]
                if args.tokenizer:
                    cmd.extend(["--tokenizer", args.tokenizer])
                for i, b in enumerate(backends):
                    cmd.extend(["-b", f"e{i}={b}"])
                session.start(f"dio_{strat}_s{seed}", cmd)
                url = f"http://127.0.0.1:{port}"
                if not hetero.wait_url(url + "/healthz", timeout=90):
                    rows.append({"seed": seed, "error": "start_failed"})
                    continue
                row = hetero.run_load(
                    url, args.model, args.requests_per_seed, args.max_tokens, 2000 + seed
                )
                row["seed"] = seed
                row["strategy"] = strat
                rows.append(row)
                if session.handles:
                    hetero.kill_proc(session.handles.pop())
                time.sleep(0.4)

            def agg(field: str):
                xs = [float(r[field]) for r in rows if r.get(field) is not None]
                return mean_std(xs)

            mapes = []
            for r in rows:
                m = (r.get("prediction") or {}).get("mape_pct")
                if m is not None:
                    mapes.append(float(m))
            summary["strategies"][strat] = {
                "frac_e0": agg("frac_e0"),
                "p99": agg("e2e_p99_ms"),
                "mape": mean_std(mapes),
                "per_seed": rows,
            }
        # overhead always local
        summary["overhead_us"] = {s: bench_overhead_us(s) for s in strategies}
        summary["status"] = "ok"
    except Exception as e:
        summary["status"] = "error"
        summary["error"] = str(e)
        import traceback
        traceback.print_exc()
    finally:
        session.cleanup()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    ap.add_argument("--out", default=str(ROOT / "results_rls_headtohead"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=200, help="sim requests/seed")
    ap.add_argument("--strategies", default="nlms,rls,round_robin,least_loaded")
    # real mode
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--backends", default="")
    ap.add_argument("--requests-per-seed", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--engine-base-port", type=int, default=18000)
    ap.add_argument("--dio-base-port", type=int, default=19300)
    ap.add_argument("--proxy-port", type=int, default=18111)
    ap.add_argument("--slow-mult", type=float, default=1.0, help=">1 for Regime-C style")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    args, _ = ap.parse_known_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "sim":
        body = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "sim",
            "result": run_sim(args.seeds, args.n),
            "status": "ok",
        }
    else:
        body = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **run_real(args),
        }

    (out / "summary.json").write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")

    # paper snippets
    lines = ["# NLMS vs RLS vs RR vs LL", f"mode={body.get('mode')}", ""]
    block = body.get("result") or body
    strats = (block.get("strategies") or {})
    for name, st in strats.items():
        if not isinstance(st, dict):
            continue
        p99 = st.get("p99") or {}
        ff = st.get("frac_fast") or st.get("frac_e0") or {}
        lines.append(
            f"- **{name}**: p99={p99.get('mean', float('nan')):.1f}±{p99.get('std', 0):.1f} "
            f"frac_fast={ff.get('mean', float('nan')):.3f}±{ff.get('std', 0):.3f}"
        )
    oh = (block.get("overhead_us") or body.get("overhead_us") or {})
    if oh:
        lines.append("")
        lines.append("## pick() overhead (µs/request, local)")
        for k, v in oh.items():
            lines.append(f"- {k}: {v:.2f} µs")
    (out / "paper_snippets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 60)
    print("RLS HEAD-TO-HEAD COMPLETE")
    print(f"Status: {body.get('status')}")
    print(f"Results → {out}")
    for line in lines:
        print(line)
    return 0 if body.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
