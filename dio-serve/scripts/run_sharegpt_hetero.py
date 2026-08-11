#!/usr/bin/env python3
"""ShareGPT-style trace replay through DIO on real heterogeneous backends.

Downloads a small public ShareGPT-format sample (or uses --trace path), then
replays user turns as chat completions under nlms vs round_robin.

Usage:
  python scripts/run_sharegpt_hetero.py \\
    --backends t4=http://127.0.0.1:8000,a30=http://HOST:8000 \\
    --vram t4=16000,a30=24000 \\
    --n-prompts 80 --seeds 3 --out results_sharegpt_hetero
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
import urllib.request
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

# Small public ShareGPT-style JSON list mirrors (fallback chain)
TRACE_URLS = [
    # compact sample used by many open evals (list of {conversations:[{from,value},...]})
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
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


def wait_url(url: str, timeout: float = 60.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=3.0).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def load_sharegpt_prompts(path: Optional[Path], n: int, seed: int) -> List[str]:
    data: Any = None
    if path and path.is_file():
        log(f"loading trace {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        cache = ROOT / "results_sharegpt_hetero" / "sharegpt_cache.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.is_file() and cache.stat().st_size > 1000:
            log(f"using cache {cache}")
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            last_err = None
            for url in TRACE_URLS:
                try:
                    log(f"downloading ShareGPT sample (may be large, streaming first chunk)...")
                    # file is huge; stream and parse only first ~2MB of array elements via ijson-less hack:
                    # download limited bytes won't be valid JSON — use datasets-style: fetch with range not reliable.
                    # Instead use a tiny built-in fallback if download too big.
                    req = urllib.request.Request(url, headers={"User-Agent": "dio-regime-d/1.0"})
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        # only read first 8MB — may fail parse
                        blob = resp.read(8 * 1024 * 1024)
                    # try to find complete objects
                    text = blob.decode("utf-8", errors="ignore")
                    # trim to last full object boundary
                    if text.lstrip().startswith("["):
                        # cut at last "}," roughly
                        cut = text.rfind("},")
                        if cut > 0:
                            text = text[: cut + 1] + "]"
                    data = json.loads(text)
                    cache.write_text(json.dumps(data[: min(500, len(data))]), encoding="utf-8")
                    log(f"cached {min(500, len(data))} conversations")
                    break
                except Exception as e:
                    last_err = e
                    log(f"download failed: {e}")
            if data is None:
                log(f"using synthetic ShareGPT-like prompts (download failed: {last_err})")
                data = []

    prompts: List[str] = []
    if isinstance(data, list) and data:
        for item in data:
            conv = item.get("conversations") or item.get("conversation") or []
            for turn in conv:
                role = (turn.get("from") or turn.get("role") or "").lower()
                if role in ("human", "user"):
                    val = (turn.get("value") or turn.get("content") or "").strip()
                    if len(val) >= 20:
                        prompts.append(val[:2000])
            if len(prompts) >= n * 3:
                break
    if len(prompts) < n:
        # pad with synthetic multi-sentence "chatty" prompts
        base = [
            "Write a detailed explanation of how transformers use attention, with examples.",
            "I have a production Kubernetes cluster. Walk me through debugging high tail latency for an LLM API.",
            "Compare PostgreSQL and MySQL for an OLTP workload with heavy writes. Be specific about locks and WAL.",
            "Explain gradient checkpointing and when I should enable it for fine-tuning a 7B model.",
            "Draft a design doc for a multi-tenant inference gateway with per-tenant SLOs and fair queuing.",
        ]
        rng = random.Random(seed)
        while len(prompts) < n:
            prompts.append(base[len(prompts) % len(base)] + f" (expand point {rng.randint(1,9)}.)")
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n]


def start_dio(backends, vram, port, strategy, tokenizer, log_path: Path):
    cmd = [
        PY, "-m", "dio", "serve",
        "--host", "127.0.0.1", "--port", str(port),
        "--strategy", strategy, "--nlms-mode", "dual",
        "--slo-ms", "180000", "--admission-off", "--admission-mode", "rank_only",
        "--engine-metrics",
    ]
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    for bid, url in backends:
        cmd.extend(["-b", f"{bid}={url}"])
    for bid, _ in backends:
        cmd.extend(["--vram", str(float(vram.get(bid, 24000)))])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["DIO_METRICS_INTERVAL_S"] = "0.5"
    kwargs: Dict[str, Any] = {"stdout": f, "stderr": subprocess.STDOUT, "cwd": str(ROOT), "env": env}
    if os.name != "nt":
        kwargs["preexec_fn"] = os.setsid
    return f"http://127.0.0.1:{port}", subprocess.Popen(cmd, **kwargs)


def run_strategy(backends, vram, strategy, prompts, model, tokenizer, max_tokens, port, log_path):
    url, proc = start_dio(backends, vram, port, strategy, tokenizer, log_path)
    e2e: List[float] = []
    routes: Dict[str, int] = {}
    ok = fail = 0
    try:
        if not wait_url(url + "/healthz", 120):
            return {"error": "gateway_failed"}
        with httpx.Client(timeout=600.0) as client:
            for i, prompt in enumerate(prompts):
                t0 = time.perf_counter()
                try:
                    r = client.post(
                        f"{url}/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": 0.0,
                        },
                        headers={"X-DIO-Tier": "small"},
                    )
                    ms = (time.perf_counter() - t0) * 1000
                    if r.status_code == 200:
                        ok += 1
                        e2e.append(ms)
                        wid = r.headers.get("X-DIO-Backend") or "unknown"
                        routes[wid] = routes.get(wid, 0) + 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1
                if (i + 1) % 20 == 0:
                    log(f"  {strategy} {i+1}/{len(prompts)} ok={ok}")
        e2e_s = sorted(e2e)
        def pct(p):
            if not e2e_s:
                return None
            k = (p / 100) * (len(e2e_s) - 1)
            lo, hi = int(k), min(len(e2e_s) - 1, int(k) + 1)
            return e2e_s[lo] * (1 - (k - lo)) + e2e_s[hi] * (k - lo)
        total = sum(routes.values()) or 1
        return {
            "ok": ok, "fail": fail, "n": len(prompts),
            "routes": routes,
            "route_frac": {k: v / total for k, v in routes.items()},
            "p50_ms": pct(50), "p95_ms": pct(95), "p99_ms": pct(99),
            "mean_ms": statistics.mean(e2e) if e2e else None,
        }
    finally:
        kill_proc(proc)
        time.sleep(0.4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", required=True)
    ap.add_argument("--vram", default="t4=16000,a30=24000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n-prompts", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--strategies", default="nlms,round_robin")
    ap.add_argument("--trace", default="")
    ap.add_argument("--out", default=str(ROOT / "results_sharegpt_hetero"))
    ap.add_argument("--dio-base-port", type=int, default=19600)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backends = parse_kv(args.backends)
    vram = {k: float(v) for k, v in parse_kv(args.vram)}
    for b, u in backends:
        if not wait_url(u.rstrip("/") + "/v1/models", 60):
            log(f"ERROR backend {b}"); return 1
        log(f"backend {b} OK")

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "run_sharegpt_hetero.py",
        "args": vars(args),
        "backends": [{"id": b, "url": u} for b, u in backends],
        "cells": {},
    }
    for si, strat in enumerate(strategies):
        per = []
        for seed in range(args.seeds):
            prompts = load_sharegpt_prompts(
                Path(args.trace) if args.trace else None,
                args.n_prompts, seed=1000 + seed,
            )
            log(f"== {strat} seed={seed} n={len(prompts)} ==")
            row = run_strategy(
                backends, vram, strat, prompts, args.model, args.tokenizer,
                args.max_tokens, args.dio_base_port + si * 20 + seed,
                out / "logs" / f"sg_{strat}_s{seed}.log",
            )
            row["seed"] = seed
            per.append(row)
            log(f"  {strat} s{seed}: p99={row.get('p99_ms')} frac={row.get('route_frac')} ok={row.get('ok')}")
        good = [r for r in per if r.get("p99_ms") is not None]
        p99s = [float(r["p99_ms"]) for r in good]
        summary["cells"][strat] = {
            "per_seed": per,
            "p99_mean": statistics.mean(p99s) if p99s else None,
            "p99_std": statistics.stdev(p99s) if len(p99s) > 1 else 0.0,
            "seeds_ok": len(good),
        }
    # improvement nlms vs rr if both present
    if "nlms" in summary["cells"] and "round_robin" in summary["cells"]:
        a = summary["cells"]["nlms"].get("p99_mean")
        b = summary["cells"]["round_robin"].get("p99_mean")
        if a and b:
            summary["p99_improvement_nlms_vs_rr_pct"] = (b - a) / b * 100.0
    summary["status"] = "ok"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "paper_snippets.md").write_text(
        "# ShareGPT-style hetero replay\n\n"
        + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                      for k, v in summary["cells"].items()}, indent=2)
        + f"\n\nimprovement: {summary.get('p99_improvement_nlms_vs_rr_pct')}\n",
        encoding="utf-8",
    )
    log(f"DONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
