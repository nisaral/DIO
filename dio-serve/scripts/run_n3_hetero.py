#!/usr/bin/env python3
"""3-worker hetero: two slow T4s + one fast A30 (or any 3 backends).

Expects backends like:
  t4a=http://...,t4b=http://...,a30=http://...

Runs short D1-style head-to-head (nlms, rls, rr) with mixed max_tokens.
Reports route mass to the fast SKU (id containing 'a30' or fastest by name).

Usage:
  python scripts/run_n3_hetero.py \\
    --backends t4a=http://127.0.0.1:8000,t4b=http://T4B:8000,a30=http://A30:8000 \\
    --vram t4a=16000,t4b=16000,a30=24000 \\
    --seeds 3 --requests-per-seed 30 --out results_n3_hetero
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
    sys.exit("pip install httpx")

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PROMPTS = [
    "Summarize causes of the industrial revolution.",
    "Explain B-tree indexes.",
    "TCP vs UDP tradeoffs.",
    "What is gradient descent?",
    "KV-cache reuse in transformers.",
    "Why batching helps GPU inference.",
    "Paged attention memory management.",
    "Optimistic vs pessimistic concurrency.",
]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_kv(spec: str) -> List[Tuple[str, str]]:
    out = []
    for i, raw in enumerate(x.strip() for x in spec.split(",")):
        if not raw:
            continue
        if "=" in raw and not raw.startswith("http"):
            k, v = raw.split("=", 1)
            out.append((k.strip(), v.strip()))
        else:
            out.append((f"e{i}", raw))
    return out


def kill_proc(p: Optional[subprocess.Popen]) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        else:
            p.terminate()
        p.wait(timeout=8)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def wait_url(url: str, t: float = 90.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            if httpx.get(url, timeout=3).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_dio(backends, vram, port, strategy, tokenizer, logf: Path):
    cmd = [PY, "-m", "dio", "serve", "--host", "127.0.0.1", "--port", str(port),
           "--strategy", strategy, "--nlms-mode", "dual", "--slo-ms", "180000",
           "--admission-off", "--admission-mode", "rank_only", "--engine-metrics"]
    if tokenizer:
        cmd += ["--tokenizer", tokenizer]
    for bid, url in backends:
        cmd += ["-b", f"{bid}={url}"]
    for bid, _ in backends:
        cmd += ["--vram", str(float(vram.get(bid, 24000)))]
    logf.parent.mkdir(parents=True, exist_ok=True)
    f = open(logf, "w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["DIO_METRICS_INTERVAL_S"] = "0.5"
    kw: Dict[str, Any] = {"stdout": f, "stderr": subprocess.STDOUT, "cwd": str(ROOT), "env": env}
    if os.name != "nt":
        kw["preexec_fn"] = os.setsid
    return f"http://127.0.0.1:{port}", subprocess.Popen(cmd, **kw)


def workload(n, seed, choices):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        pad = "More detail. " * rng.randint(0, 5)
        out.append({
            "prompt": f"{PROMPTS[i % len(PROMPTS)]} {pad}(s={seed},i={i})",
            "max_tokens": rng.choice(choices),
        })
    return out


def run_one(backends, vram, strategy, seed, n, model, tokenizer, port, logf):
    url, proc = start_dio(backends, vram, port, strategy, tokenizer, logf)
    try:
        if not wait_url(url + "/healthz", 120):
            return {"seed": seed, "error": "gw"}
        items = workload(n, 2000 + seed, [32, 64, 128, 256])
        e2e, routes, ok, fail = [], {}, 0, 0
        with httpx.Client(timeout=600) as c:
            for it in items:
                t0 = time.perf_counter()
                try:
                    r = c.post(f"{url}/v1/chat/completions", json={
                        "model": model,
                        "messages": [{"role": "user", "content": it["prompt"]}],
                        "max_tokens": it["max_tokens"], "temperature": 0.0,
                    }, headers={"X-DIO-Tier": "small"})
                    ms = (time.perf_counter() - t0) * 1000
                    if r.status_code == 200:
                        ok += 1
                        e2e.append(ms)
                        w = r.headers.get("X-DIO-Backend") or "?"
                        routes[w] = routes.get(w, 0) + 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1
        e2e_s = sorted(e2e)
        def pct(p):
            if not e2e_s:
                return None
            k = (p/100)*(len(e2e_s)-1)
            lo, hi = int(k), min(len(e2e_s)-1, int(k)+1)
            return e2e_s[lo]*(1-(k-lo))+e2e_s[hi]*(k-lo)
        tot = sum(routes.values()) or 1
        return {
            "seed": seed, "ok": ok, "fail": fail, "routes": routes,
            "route_frac": {k: v/tot for k, v in routes.items()},
            "p50_ms": pct(50), "p99_ms": pct(99),
            "mean_ms": statistics.mean(e2e) if e2e else None,
        }
    finally:
        kill_proc(proc)
        time.sleep(0.4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", required=True)
    ap.add_argument("--vram", default="")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--requests-per-seed", type=int, default=30)
    ap.add_argument("--strategies", default="nlms,rls,round_robin")
    ap.add_argument("--fast-id", default="a30", help="worker id expected to get majority under NLMS")
    ap.add_argument("--out", default=str(ROOT / "results_n3_hetero"))
    ap.add_argument("--dio-base-port", type=int, default=19700)
    args = ap.parse_args()

    backends = parse_kv(args.backends)
    if len(backends) < 3:
        log("ERROR: need >=3 backends for N=3 study")
        return 2
    vram = {k: float(v) for k, v in parse_kv(args.vram)} if args.vram else {b: 24000.0 for b, _ in backends}
    for b, u in backends:
        if not wait_url(u.rstrip("/") + "/v1/models", 90):
            log(f"ERROR down {b} {u}"); return 1
        log(f"OK {b}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    strats = [s.strip() for s in args.strategies.split(",") if s.strip()]
    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "run_n3_hetero.py",
        "args": vars(args),
        "backends": [{"id": b, "url": u, "vram": vram.get(b)} for b, u in backends],
        "cells": {},
    }
    for si, strat in enumerate(strats):
        rows = []
        for seed in range(args.seeds):
            log(f"== {strat} seed={seed} ==")
            row = run_one(
                backends, vram, strat, seed, args.requests_per_seed,
                args.model, args.tokenizer,
                args.dio_base_port + si * 30 + seed,
                out / "logs" / f"n3_{strat}_s{seed}.log",
            )
            rows.append(row)
            log(f"  {strat} s{seed}: p99={row.get('p99_ms')} frac={row.get('route_frac')}")
        good = [r for r in rows if r.get("p99_ms") is not None]
        p99s = [float(r["p99_ms"]) for r in good]
        fast_fracs = [float((r.get("route_frac") or {}).get(args.fast_id) or 0) for r in good]
        summary["cells"][strat] = {
            "per_seed": rows,
            "p99_mean": statistics.mean(p99s) if p99s else None,
            "p99_std": statistics.stdev(p99s) if len(p99s) > 1 else 0.0,
            f"{args.fast_id}_frac_mean": statistics.mean(fast_fracs) if fast_fracs else None,
            "seeds_ok": len(good),
        }
    summary["status"] = "ok"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "paper_snippets.md").write_text(
        "# N=3 hetero (2×slow + 1×fast)\n\n"
        + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                      for k, v in summary["cells"].items()}, indent=2),
        encoding="utf-8",
    )
    log(f"DONE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
