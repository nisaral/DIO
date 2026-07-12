#!/usr/bin/env python3
"""
==============================================================================
DIO PAPER EXPERIMENTS — one script, library methods only
==============================================================================

Runs the full novelty / publishability matrix used in the DIO paper by calling
the installed ``dio`` package (Scheduler, AblationFlags, metrics, optional
HTTP gateway with mock engines for CI).

This is NOT production traffic. Production = real vLLM URLs via ``dio serve``.
This script validates algorithms and generates paper tables/figures anywhere
(laptop, CI, cloud CPU).

Usage
-----
    cd dio-serve
    pip install -e .
    python scripts/run_paper_experiments.py              # full
    python scripts/run_paper_experiments.py --quick       # smoke
    python scripts/run_paper_experiments.py --out ./paper_results

With real engines (optional load test after suite)::

    python scripts/run_paper_experiments.py --real-backends \\
        http://127.0.0.1:8000,http://127.0.0.1:8001 --real-requests 50

Outputs
-------
    <out>/summary.json   all metrics
    <out>/tables.json    flat rows for LaTeX
    <out>/figures/       PNG plots if matplotlib available
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Re-use the comprehensive suite implementation
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_publishable_suite import (  # noqa: E402
    exp_G1_gateway_http,
    exp_L0_library_overhead,
    exp_N1_dual_vs_single,
    exp_N2_admission,
    exp_N2b_vram_hard,
    exp_N3_tiers,
    exp_N4_ablations,
    exp_N5_scale,
    log,
    try_plots,
)


def exp_real_backends(urls: list[str], n_requests: int) -> dict:
    """Optional: hit REAL OpenAI-compatible engines through DIOGateway."""
    import asyncio
    import time

    import httpx
    import uvicorn
    from dio import Backend, DIOGateway

    async def _run() -> dict:
        backends = [
            Backend(id=f"real{i}", base_url=u.strip(), api_style="openai")
            for i, u in enumerate(urls)
            if u.strip()
        ]
        if not backends:
            return {"name": "REAL", "error": "no backends"}
        port = 18111
        gw = DIOGateway(
            backends=backends,
            strategy="nlms",
            nlms_mode="dual",
            admission_off=True,
            slo_ms=120_000,
            port=port,
        )
        cfg = uvicorn.Config(gw.app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(cfg)
        task = asyncio.create_task(server.serve())
        async with httpx.AsyncClient(timeout=180.0) as client:
            for _ in range(60):
                try:
                    if (await client.get(f"http://127.0.0.1:{port}/healthz")).status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.2)

            ok = 0
            lats = []
            routes: dict[str, int] = {}
            for i in range(n_requests):
                t0 = time.perf_counter()
                r = await client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={
                        "model": backends[0].model or "default",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Paper probe {i}: say OK in one word.",
                            }
                        ],
                        "max_tokens": 16,
                    },
                )
                lats.append((time.perf_counter() - t0) * 1000)
                if r.status_code == 200:
                    ok += 1
                    b = r.headers.get("X-DIO-Backend", "?")
                    routes[b] = routes.get(b, 0) + 1
                else:
                    log(f"  real req {i}: HTTP {r.status_code} {r.text[:120]}")
        server.should_exit = True
        await task
        lats_sorted = sorted(lats)
        p99 = lats_sorted[int(0.99 * (len(lats_sorted) - 1))] if lats_sorted else None
        m = gw.scheduler.metrics()
        return {
            "name": "REAL_engines",
            "backends": [b.base_url for b in backends],
            "n_ok": ok,
            "n_req": n_requests,
            "p99_ms": p99,
            "routes": routes,
            "mape_pct": m["prediction"]["mape_pct"],
            "workers": m["workers"],
        }

    log("=== REAL engine smoke (production path) ===")
    out = asyncio.run(_run())
    log(f"  real: ok={out.get('n_ok')}/{out.get('n_req')} p99={out.get('p99_ms')} routes={out.get('routes')}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="DIO paper experiments via dio library")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-gateway", action="store_true")
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "results_paper"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--real-backends",
        type=str,
        default="",
        help="Comma-separated real engine base URLs for production-path smoke",
    )
    ap.add_argument("--real-requests", type=int, default=30)
    args = ap.parse_args()

    import random
    from datetime import datetime, timezone

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 120 if args.quick else 500
    n_scale_w = 16 if args.quick else 64
    n_scale_r = 200 if args.quick else 800

    import dio

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "run_paper_experiments.py",
        "dio_version": dio.__version__,
        "note": "Algorithmic paper suite via library; use --real-backends for live engines",
        "library_methods_used": [
            "Scheduler(strategy, dual, ablation, slo_ms, admission_off)",
            "Scheduler.register / pick / feedback / metrics / set_vram / set_healthy",
            "AblationFlags",
            "DIOGateway + Backend (G1 + optional REAL)",
            "MockBackendServer (CI only for G1)",
        ],
    }

    try:
        summary["L0_overhead"] = exp_L0_library_overhead(5000 if args.quick else 25000)
        summary["N1_dual_vs_single"] = exp_N1_dual_vs_single(n)
        summary["N2_admission"] = exp_N2_admission(n)
        summary["N2b_vram"] = exp_N2b_vram_hard(40 if args.quick else 100)
        summary["N3_tiers"] = exp_N3_tiers(n)
        summary["N4_ablations"] = exp_N4_ablations(max(100, n // 2))
        summary["N5_scale"] = exp_N5_scale(n_scale_w, n_scale_r)
        if not args.skip_gateway:
            import asyncio

            summary["G1_gateway"] = asyncio.run(
                exp_G1_gateway_http(5.0 if args.quick else 12.0)
            )
        if args.real_backends.strip():
            urls = [u.strip() for u in args.real_backends.split(",") if u.strip()]
            summary["REAL"] = exp_real_backends(urls, args.real_requests)
    except Exception as e:
        summary["error"] = str(e)
        log(f"FATAL: {e}")
        import traceback

        traceback.print_exc()

    rows = []
    for key, block in summary.items():
        if key in ("generated_at", "script", "dio_version", "note", "library_methods_used", "error"):
            continue
        if isinstance(block, list):
            for r in block:
                rows.append({"suite": key, **{k: r.get(k) for k in r if k != "workers"}})
        elif isinstance(block, dict):
            rows.append({"suite": key, **{k: block.get(k) for k in block if k != "workers"}})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "tables.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    # figures dir next to out
    from run_publishable_suite import FIGS
    import run_publishable_suite as rps

    rps.FIGS = out_dir / "figures"
    rps.FIGS.mkdir(parents=True, exist_ok=True)
    # map keys for plotter
    plot_summary = {
        "N1": summary.get("N1_dual_vs_single"),
        "N2": summary.get("N2_admission"),
        "N4": summary.get("N4_ablations"),
    }
    try_plots(plot_summary)

    print("\n" + "=" * 64)
    print("PAPER EXPERIMENTS COMPLETE")
    print("=" * 64)
    print(f"dio {dio.__version__}")
    if "L0_overhead" in summary:
        print(f"L0 pick+feedback: {summary['L0_overhead']['pick_feedback_us']:.1f} µs")
    for r in summary.get("N1_dual_vs_single") or []:
        print(f"N1 {r.get('nlms_mode')}: p99={r.get('p99_ms')} mape={r.get('prediction',{}).get('mape_pct')}")
    for r in summary.get("N4_ablations") or []:
        if r.get("strategy") in ("nlms", "round_robin") and r.get("ablation") == "full":
            print(f"N4 {r.get('strategy')}/full: p99={r.get('p99_ms')}")
    if "REAL" in summary:
        print(f"REAL engines: {summary['REAL']}")
    print(f"\nResults → {out_dir}")
    return 0 if "error" not in summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
