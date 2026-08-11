#!/usr/bin/env python3
"""Post-Regime-D high-value follow-ups on a live T4+A30 pair.

1) SKU discovery time: cold-start NLMS, count requests until learned slope
   ordering matches OLS ground truth (t4 slower than a30).
2) Extra D1 seeds for nlms+rls only (seed offset so we don't redo 0..4).
3) Calibration table: learned vs OLS from an existing summary.json + live
   discovery curve written to JSON for paper plots.

Usage (engines already up):
  python scripts/run_post_d_priority.py \\
    --backends t4=http://127.0.0.1:8000,a30=http://HOST:8000 \\
    --vram t4=16000,a30=24000 \\
    --prior-summary /root/results_regime_d/summary.json \\
    --out /root/results_post_d_priority
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    print("pip install httpx", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# reuse helpers from regime d by light duplication (standalone, no import cycle)
PROMPT_SEEDS = [
    "Summarize the causes of the industrial revolution.",
    "Explain how a B-tree index speeds up lookups.",
    "Describe the tradeoffs between TCP and UDP.",
    "What is gradient descent and why does it converge?",
    "Compare optimistic and pessimistic concurrency control.",
    "Explain KV-cache reuse in transformer inference.",
    "Why does batching improve GPU utilization?",
    "Describe how paged attention manages memory.",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_kv(spec: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for i, raw in enumerate(x.strip() for x in spec.split(",")):
        if not raw:
            continue
        if "=" in raw and not raw.startswith("http"):
            k, v = raw.split("=", 1)
            out.append((k.strip(), v.strip()))
        else:
            out.append((f"e{i}", raw))
    return out


def kill_proc(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=8)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def wait_url(url: str, timeout: float = 120.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=3.0).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def start_dio(
    backends: List[Tuple[str, str]],
    vram: Dict[str, float],
    port: int,
    strategy: str,
    tokenizer: str,
    log_path: Path,
) -> Tuple[str, subprocess.Popen]:
    cmd = [
        PY, "-m", "dio", "serve",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--strategy", strategy,
        "--nlms-mode", "dual",
        "--slo-ms", "180000",
        "--admission-off",
        "--admission-mode", "rank_only",
        "--engine-metrics",
    ]
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    for bid, url in backends:
        cmd.extend(["-b", f"{bid}={url}"])
    for bid, _ in backends:
        cmd.extend(["--vram", str(float(vram.get(bid, 24000.0)))])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["DIO_METRICS_INTERVAL_S"] = "0.5"
    kwargs: Dict[str, Any] = {
        "stdout": f, "stderr": subprocess.STDOUT, "cwd": str(ROOT), "env": env,
    }
    if os.name != "nt":
        kwargs["preexec_fn"] = os.setsid
    p = subprocess.Popen(cmd, **kwargs)
    return f"http://127.0.0.1:{port}", p


def build_workload(n: int, seed: int, max_tokens_choices: List[int]) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        body = PROMPT_SEEDS[i % len(PROMPT_SEEDS)]
        pad = "Provide relevant background detail. " * rng.randint(0, 6)
        items.append({
            "prompt": f"{body} {pad}(seed={seed} i={i})".strip(),
            "max_tokens": rng.choice(max_tokens_choices),
        })
    return items


def post_one(client: httpx.Client, base: str, model: str, item: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = client.post(
        f"{base.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": item["max_tokens"],
            "temperature": 0.0,
        },
        headers={"X-DIO-Tier": "small"},
    )
    ms = (time.perf_counter() - t0) * 1000.0
    row: Dict[str, Any] = {"ok": r.status_code == 200, "e2e_ms": ms, "status": r.status_code}
    if r.status_code == 200:
        row["worker"] = r.headers.get("X-DIO-Backend") or "unknown"
        try:
            j = r.json()
            dec = (j.get("dio") or {}).get("decision") or {}
            row["tokens_feature"] = dec.get("tokens")
        except Exception:
            pass
    return row


def get_slopes(client: httpx.Client, base: str) -> Dict[str, Any]:
    try:
        m = client.get(f"{base.rstrip('/')}/debug/metrics", timeout=20.0).json()
    except Exception as e:
        return {"error": str(e)}
    out = {}
    for wid, w in (m.get("workers") or {}).items():
        if isinstance(w, dict):
            # dual mode: prefer slow_slope (decode) when present, else fast
            s = w.get("slow_slope")
            if s is None:
                s = w.get("fast_slope")
            out[wid] = {
                "slope": s,
                "fast_slope": w.get("fast_slope"),
                "slow_slope": w.get("slow_slope"),
                "intercept": w.get("intercept"),
                "updates": w.get("updates"),
            }
    return out


def discovery_run(
    backends: List[Tuple[str, str]],
    vram: Dict[str, float],
    *,
    model: str,
    tokenizer: str,
    out: Path,
    max_requests: int,
    margin_ratio: float,
    port: int,
) -> Dict[str, Any]:
    """Cold NLMS: how many reqs until t4_slope > a30_slope * (1+margin)."""
    log_path = out / "logs" / "discovery_gateway.log"
    url, proc = start_dio(backends, vram, port, "nlms", tokenizer, log_path)
    curve: List[Dict[str, Any]] = []
    discovered_at: Optional[int] = None
    try:
        if not wait_url(url + "/healthz", timeout=120):
            return {"error": "gateway_start_failed"}
        workload = build_workload(max_requests, seed=4242, max_tokens_choices=[32, 64, 128, 256])
        with httpx.Client(timeout=600.0) as client:
            for i, item in enumerate(workload):
                row = post_one(client, url, model, item)
                slopes = get_slopes(client, url)
                t4 = (slopes.get("t4") or {}).get("slope")
                a30 = (slopes.get("a30") or {}).get("slope")
                order_ok = (
                    t4 is not None and a30 is not None
                    and a30 > 0 and t4 > a30 * (1.0 + margin_ratio)
                )
                point = {
                    "n": i + 1,
                    "ok": row.get("ok"),
                    "worker": row.get("worker"),
                    "e2e_ms": row.get("e2e_ms"),
                    "t4_slope": t4,
                    "a30_slope": a30,
                    "order_ok": order_ok,
                    "updates_t4": (slopes.get("t4") or {}).get("updates"),
                    "updates_a30": (slopes.get("a30") or {}).get("updates"),
                }
                curve.append(point)
                if order_ok and discovered_at is None:
                    discovered_at = i + 1
                    log(f"DISCOVERY at n={discovered_at}: t4={t4:.4f} a30={a30:.4f}")
                    # keep a few more for stability window
                if discovered_at is not None and (i + 1) >= discovered_at + 10:
                    break
                if (i + 1) % 10 == 0:
                    log(f"discovery n={i+1} t4={t4} a30={a30} order={order_ok}")
        # stability: first n where next 5 also ok
        stable_at = None
        for i in range(len(curve)):
            if not curve[i].get("order_ok"):
                continue
            window = curve[i:i + 5]
            if len(window) >= 5 and all(p.get("order_ok") for p in window):
                stable_at = curve[i]["n"]
                break
        return {
            "max_requests": max_requests,
            "margin_ratio": margin_ratio,
            "discovered_at_first": discovered_at,
            "discovered_at_stable5": stable_at,
            "final_slopes": curve[-1] if curve else None,
            "curve": curve,
            "claim": (
                f"NLMS correctly ranks t4 slower than a30 within "
                f"{stable_at or discovered_at or 'N/A'} requests "
                f"(margin={margin_ratio})"
            ),
        }
    finally:
        kill_proc(proc)
        time.sleep(0.5)


def extra_d1_seeds(
    backends: List[Tuple[str, str]],
    vram: Dict[str, float],
    *,
    model: str,
    tokenizer: str,
    out: Path,
    strategies: List[str],
    n_seeds: int,
    seed_offset: int,
    requests_per_seed: int,
    base_port: int,
) -> Dict[str, Any]:
    """Additional D1 seeds with offset workloads (does not clobber 0..4)."""
    from collections import defaultdict
    rows: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}
    max_tokens = [32, 64, 128, 256]
    for si, strat in enumerate(strategies):
        for j in range(n_seeds):
            seed = seed_offset + j
            workload = build_workload(requests_per_seed, 1000 + seed, max_tokens)
            port = base_port + si * 50 + j
            log_path = out / "logs" / f"extra_d1_{strat}_s{seed}.log"
            url, proc = start_dio(backends, vram, port, strat, tokenizer, log_path)
            try:
                if not wait_url(url + "/healthz", timeout=120):
                    rows[strat].append({"seed": seed, "error": "gateway_start_failed"})
                    continue
                e2e: List[float] = []
                routes: Dict[str, int] = {}
                ok = fail = 0
                with httpx.Client(timeout=600.0) as client:
                    for item in workload:
                        row = post_one(client, url, model, item)
                        if row.get("ok"):
                            ok += 1
                            e2e.append(float(row["e2e_ms"]))
                            wid = row.get("worker") or "unknown"
                            routes[wid] = routes.get(wid, 0) + 1
                        else:
                            fail += 1
                e2e_s = sorted(e2e)
                def pct(p: float) -> Optional[float]:
                    if not e2e_s:
                        return None
                    k = (p / 100.0) * (len(e2e_s) - 1)
                    lo, hi = int(k), min(len(e2e_s) - 1, int(k) + 1)
                    return e2e_s[lo] * (1 - (k - lo)) + e2e_s[hi] * (k - lo)
                total = sum(routes.values()) or 1
                rec = {
                    "seed": seed,
                    "ok": ok,
                    "fail": fail,
                    "routes": routes,
                    "route_frac": {k: v / total for k, v in routes.items()},
                    "e2e_p50_ms": pct(50),
                    "e2e_p99_ms": pct(99),
                    "e2e_mean_ms": statistics.mean(e2e) if e2e else None,
                }
                rows[strat].append(rec)
                log(f"extra d1 {strat} seed={seed}: p99={rec['e2e_p99_ms']} frac={rec['route_frac']} ok={ok}/{ok+fail}")
            finally:
                kill_proc(proc)
                time.sleep(0.4)
    # aggregate
    summary: Dict[str, Any] = {"seed_offset": seed_offset, "n_seeds": n_seeds, "per_seed": rows, "agg": {}}
    for strat, lst in rows.items():
        good = [r for r in lst if not r.get("error") and r.get("e2e_p99_ms") is not None]
        if not good:
            continue
        p99s = [float(r["e2e_p99_ms"]) for r in good]
        frac_a = [float((r.get("route_frac") or {}).get("a30") or 0.0) for r in good]
        summary["agg"][strat] = {
            "p99_mean": statistics.mean(p99s),
            "p99_std": statistics.stdev(p99s) if len(p99s) > 1 else 0.0,
            "a30_frac_mean": statistics.mean(frac_a),
            "seeds_ok": len(good),
        }
    return summary


def calibration_from_prior(prior_path: Path) -> Dict[str, Any]:
    if not prior_path.is_file():
        return {"error": f"missing {prior_path}"}
    s = json.loads(prior_path.read_text(encoding="utf-8"))
    d2 = s.get("D2_slope_identifiability") or {}
    d3 = s.get("D3_max_tokens_sensitivity") or {}
    d1 = s.get("D1_strategy_head_to_head") or {}
    cal = {
        "source": str(prior_path),
        "D2": {
            "slopes_by_worker": d2.get("slopes_by_worker"),
            "slope_ratio": d2.get("slope_ratio"),
            "note": "learned slopes are NLMS (shrunk); OLS is offline ground truth on same RR traffic",
        },
        "D3_learned_slopes_by_max_tokens": {},
        "D1_headline": {},
    }
    for L, c in (d3.get("cells") or {}).items():
        cal["D3_learned_slopes_by_max_tokens"][str(L)] = {
            "learned_slope": c.get("learned_slope"),
            "p99": c.get("p99"),
            "ols_degenerate_seeds": c.get("ols_degenerate_seeds"),
        }
    for strat in ("nlms", "rls", "round_robin", "least_loaded"):
        row = d1.get(strat)
        if row:
            cal["D1_headline"][strat] = {
                "p99": row.get("p99"),
                "route_frac": row.get("route_frac"),
                "seeds_ok": row.get("seeds_ok"),
            }
    for key in ("p99_improvement_vs_rr", "p99_improvement_vs_rr_rls", "p99_improvement_vs_rr_least_loaded"):
        if key in d1:
            cal["D1_headline"][key] = d1[key]
    # ratio consistency: D2 OLS ~2.8x; learned ratio should preserve order
    ratio = d2.get("slope_ratio") or {}
    cal["ordering_preserved"] = None
    if ratio.get("learned_ratio") is not None and ratio.get("ols_ratio") is not None:
        # paper stores a30/t4 so smaller = a30 faster
        cal["ordering_preserved"] = (
            float(ratio["learned_ratio"]) < 1.0 and float(ratio["ols_ratio"]) < 1.0
        )
    return cal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", required=True)
    ap.add_argument("--vram", default="t4=16000,a30=24000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--prior-summary", default="")
    ap.add_argument("--out", default=str(ROOT / "results_post_d_priority"))
    ap.add_argument("--discovery-max", type=int, default=80)
    ap.add_argument("--discovery-margin", type=float, default=0.15,
                    help="require t4_slope > a30_slope * (1+margin)")
    ap.add_argument("--extra-seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=100)
    ap.add_argument("--requests-per-seed", type=int, default=40)
    ap.add_argument("--skip-discovery", action="store_true")
    ap.add_argument("--skip-extra-seeds", action="store_true")
    ap.add_argument("--dio-base-port", type=int, default=19400)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    backends = parse_kv(args.backends)
    vram = {k: float(v) for k, v in parse_kv(args.vram)}
    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "run_post_d_priority.py",
        "args": vars(args),
        "backends": [{"id": b, "url": u, "vram_mb": vram.get(b, 24000)} for b, u in backends],
    }

    # engines live?
    for bid, url in backends:
        if not wait_url(url.rstrip("/") + "/v1/models", timeout=60):
            log(f"ERROR: backend down {bid} {url}")
            summary["status"] = "error"
            summary["error"] = f"backend {bid}"
            (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return 1
        log(f"backend {bid} OK")

    if args.prior_summary:
        summary["calibration_from_regime_d"] = calibration_from_prior(Path(args.prior_summary))
        log(f"calibration loaded; ordering_preserved={summary['calibration_from_regime_d'].get('ordering_preserved')}")

    if not args.skip_discovery:
        log("== SKU discovery time ==")
        summary["sku_discovery"] = discovery_run(
            backends, vram,
            model=args.model, tokenizer=args.tokenizer, out=out,
            max_requests=args.discovery_max, margin_ratio=args.discovery_margin,
            port=args.dio_base_port,
        )
        log(f"discovery: {summary['sku_discovery'].get('claim')}")

    if not args.skip_extra_seeds:
        log("== Extra D1 seeds (nlms,rls) ==")
        summary["extra_d1_seeds"] = extra_d1_seeds(
            backends, vram,
            model=args.model, tokenizer=args.tokenizer, out=out,
            strategies=["nlms", "rls"],
            n_seeds=args.extra_seeds,
            seed_offset=args.seed_offset,
            requests_per_seed=args.requests_per_seed,
            base_port=args.dio_base_port + 100,
        )

    summary["status"] = "ok"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    # short markdown
    lines = [
        "# Post-D priority results",
        f"Generated: {summary['generated_at']}",
        "",
        "## Calibration (from Regime D summary)",
        json.dumps(summary.get("calibration_from_regime_d"), indent=2, default=str)[:3000],
        "",
        "## SKU discovery",
        json.dumps({k: v for k, v in (summary.get("sku_discovery") or {}).items() if k != "curve"}, indent=2),
        "",
        "## Extra D1 seeds agg",
        json.dumps((summary.get("extra_d1_seeds") or {}).get("agg"), indent=2),
        "",
    ]
    (out / "paper_snippets.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"DONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
