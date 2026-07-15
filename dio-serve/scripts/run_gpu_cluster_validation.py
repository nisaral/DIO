#!/usr/bin/env python3
"""
==============================================================================
DIO GPU CLUSTER VALIDATION — GRAND SCRIPT
==============================================================================
One entry point for all *real-hardware / real-engine* tests needed before paper
camera-ready and production claims. Uses the installed ``dio`` library.

What it covers
--------------
  P0  Environment probe (CUDA, torch, GPUs, dio version)
  G1  Single-backend smoke (real OpenAI-compatible engine)
  G2  Multi-seed strategy matrix (NLMS / RR / LL / RLS / STATIC) × seeds
  G3  Dual-backend heterogeneity (real engines + optional delay-proxy throttle)
  G4  Dual vs single NLMS mode (multi-seed MAPE / p99)
  G5  Admission ON vs OFF under load
  G6  TTFT vs end-to-end instrumentation (if engine returns usage/timing)
  G7  Optional paper microbench (T2 multiseed, no GPU) for completeness

Engine modes
------------
  --engine-mode vllm     Start vLLM OpenAI servers (preferred on cluster)
  --engine-mode hf       Start scripts/real_engine_server.py (transformers)
  --engine-mode external Use already-running backends (no start/stop)

Heterogeneity (important)
-------------------------
  Homogeneous dual-GPU (G2) keeps both backends raw.
  G3 optionally wraps the *second* backend with latency_delay_proxy.py when
  --hetero-slow-mult > 1 so one peer looks 2× slower while still serving
  real tokens (T2-style real-GPU hetero). Prefer --hetero-seeds 5.

Quick start (cluster)
---------------------
  cd dio-serve
  pip install -e .
  # 2× GPU, small model, 3 seeds (paper minimum)
  python scripts/run_gpu_cluster_validation.py \\
      --engine-mode vllm \\
      --model meta-llama/Llama-3.2-3B-Instruct \\
      --gpus 0,1 \\
      --seeds 3 \\
      --requests-per-seed 40 \\
      --max-tokens 32

  # One more round: real dual-T4 hetero only (5 seeds, one throttled worker)
  python scripts/run_gpu_cluster_validation.py \\
      --engine-mode vllm \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --gpus 0,1 \\
      --skip-g2 --skip-g4 --skip-g5 \\
      --hetero-slow-mult 2.0 \\
      --hetero-seeds 5 \\
      --requests-per-seed 30 \\
      --out results_gpu_cluster_hetero

  # Engines already up
  python scripts/run_gpu_cluster_validation.py \\
      --engine-mode external \\
      --backends http://127.0.0.1:8000,http://127.0.0.1:8001 \\
      --seeds 3

  # Laptop / 4050 single GPU
  python scripts/run_gpu_cluster_validation.py \\
      --engine-mode hf \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --gpus 0 \\
      --seeds 3 --requests-per-seed 12 --max-tokens 24

Outputs
-------
  results_gpu_cluster/
    summary.json          full nested results
    tables.csv            flat table for Excel/LaTeX
    paper_snippets.md     ready-to-paste mean±std lines
    logs/                 engine + dio logs
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
ENGINE_HF = ROOT / "scripts" / "real_engine_server.py"
DELAY_PROXY = ROOT / "scripts" / "latency_delay_proxy.py"
DEFAULT_OUT = ROOT / "results_gpu_cluster"


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mean_std(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return {"mean": m, "std": sd, "ci95": ci, "n": len(xs)}


def pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[i]


def wait_url(url: str, timeout: float = 600.0, interval: float = 2.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def kill_proc(proc: Optional[subprocess.Popen]) -> None:
    if not proc or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(0.5)
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def probe_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": PY,
        "cwd": str(Path.cwd()),
        "dio_version": None,
        "torch": {},
        "nvidia_smi": None,
        "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import dio

        info["dio_version"] = dio.__version__
    except Exception as e:
        info["dio_import_error"] = str(e)
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as e:
        info["torch"] = {"error": str(e)}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0:
            info["nvidia_smi"] = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
    except Exception as e:
        info["nvidia_smi_error"] = str(e)
    return info


# ---------------------------------------------------------------------------
# Engine + DIO process management
# ---------------------------------------------------------------------------
@dataclass
class ProcHandle:
    name: str
    proc: subprocess.Popen
    log_path: Path
    url: Optional[str] = None


class ClusterSession:
    def __init__(self, out: Path):
        self.out = out
        self.logs = out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.handles: List[ProcHandle] = []

    def start(
        self,
        name: str,
        cmd: List[str],
        env: Optional[Dict[str, str]] = None,
        url: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> ProcHandle:
        log_path = self.logs / f"{name}.log"
        f = open(log_path, "w", encoding="utf-8")
        e = os.environ.copy()
        if env:
            e.update(env)
        kwargs: Dict[str, Any] = {
            "stdout": f,
            "stderr": subprocess.STDOUT,
            "cwd": cwd or str(ROOT),
            "env": e,
        }
        if os.name != "nt":
            kwargs["preexec_fn"] = os.setsid
        log(f"START {name}: {' '.join(cmd[:12])}{'...' if len(cmd)>12 else ''}")
        proc = subprocess.Popen(cmd, **kwargs)
        h = ProcHandle(name=name, proc=proc, log_path=log_path, url=url)
        self.handles.append(h)
        return h

    def cleanup(self) -> None:
        log("Cleaning up processes...")
        for h in reversed(self.handles):
            kill_proc(h.proc)
        self.handles.clear()
        time.sleep(1.0)


def start_vllm_engine(
    session: ClusterSession,
    *,
    gpu: str,
    port: int,
    model: str,
    max_model_len: int,
    gpu_mem_util: float,
    name: str,
) -> ProcHandle:
    # Prefer module entrypoint
    # Note: older vLLM used --disable-log-requests; newer uses --no-enable-log-requests.
    # Omit optional log flags for max version compatibility (Kaggle/cluster).
    cmd = [
        PY,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
    ]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    return session.start(
        name,
        cmd,
        env=env,
        url=f"http://127.0.0.1:{port}",
    )


def start_hf_engine(
    session: ClusterSession,
    *,
    gpu: Optional[str],
    port: int,
    model: str,
    latency_mult: float,
    name: str,
) -> ProcHandle:
    cmd = [
        PY,
        str(ENGINE_HF),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        model,
        "--latency-mult",
        str(latency_mult),
    ]
    env = {}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return session.start(name, cmd, env=env or None, url=f"http://127.0.0.1:{port}")


def start_delay_proxy(
    session: ClusterSession,
    *,
    upstream: str,
    port: int,
    latency_mult: float,
    name: str = "delay_proxy",
) -> ProcHandle:
    """Wrap a real OpenAI backend so e2e wall time ≈ mult × upstream time."""
    if not DELAY_PROXY.exists():
        raise FileNotFoundError(f"missing delay proxy script: {DELAY_PROXY}")
    cmd = [
        PY,
        str(DELAY_PROXY),
        "--upstream",
        upstream,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--latency-mult",
        str(latency_mult),
    ]
    return session.start(name, cmd, url=f"http://127.0.0.1:{port}")


def make_hetero_backends(
    session: ClusterSession,
    backends: List[str],
    *,
    slow_mult: float,
    proxy_port: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    For G3: keep backend[0] raw (fast), wrap backend[1] with delay proxy when mult>1.
    Homogeneous G2 should keep using the raw ``backends`` list.
    """
    meta: Dict[str, Any] = {
        "slow_mult": slow_mult,
        "raw_backends": list(backends),
        "throttled": False,
    }
    if len(backends) < 2 or slow_mult <= 1.0:
        meta["note"] = "no throttle (homogeneous or single backend)"
        return list(backends), meta
    h = start_delay_proxy(
        session,
        upstream=backends[1],
        port=proxy_port,
        latency_mult=slow_mult,
        name="delay_proxy_slow",
    )
    proxy_url = f"http://127.0.0.1:{proxy_port}"
    # proxy serves /health; engines serve /v1/models
    if not wait_url(proxy_url + "/health", timeout=60):
        log("WARN: delay proxy health check failed — G3 may be unreliable")
    meta["throttled"] = True
    meta["proxy_url"] = proxy_url
    meta["proxy_log"] = str(h.log_path)
    meta["note"] = (
        f"e1 is real engine {backends[1]} behind delay proxy ×{slow_mult} "
        "(real tokens; inflated e2e for slope-skew)"
    )
    return [backends[0], proxy_url], meta


def start_dio(
    session: ClusterSession,
    *,
    backends: List[str],
    port: int,
    strategy: str,
    nlms_mode: str = "dual",
    slo_ms: float = 120_000,
    admission_off: bool = True,
    admission_mode: str = "rank_only",
    tokenizer: str = "",
    name: str = "dio",
) -> ProcHandle:
    cmd = [
        PY,
        "-m",
        "dio",
        "serve",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--strategy",
        strategy,
        "--nlms-mode",
        nlms_mode,
        "--slo-ms",
        str(slo_ms),
        "--admission-mode",
        admission_mode,
    ]
    if admission_off:
        cmd.append("--admission-off")
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    for i, b in enumerate(backends):
        cmd.extend(["-b", f"e{i}={b}"])
    env = {
        "DIO_STRATEGY": strategy,
        "DIO_NLMS_MODE": nlms_mode,
        "DIO_SLO_MS": str(slo_ms),
        "DIO_ADMISSION_OFF": "1" if admission_off else "0",
        "DIO_ADMISSION_MODE": admission_mode,
    }
    if tokenizer:
        env["DIO_TOKENIZER_NAME"] = tokenizer
        env["DIO_USE_TOKENIZER"] = "1"
    return session.start(name, cmd, env=env, url=f"http://127.0.0.1:{port}")


# ---------------------------------------------------------------------------
# Load generation
# ---------------------------------------------------------------------------
PROMPTS = [
    "What is 2+2? Answer in one short sentence.",
    "Name three colors. Be brief.",
    "Explain gravity in one sentence.",
    "Write a haiku about rain.",
    "List two benefits of exercise.",
    "What is the capital of France? One word if possible.",
    "Summarize photosynthesis in one sentence.",
    "Say hello and ask how I am.",
]


def chat_once(
    client: httpx.Client,
    base: str,
    prompt: str,
    model: str,
    max_tokens: int,
    tier: str = "small",
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = client.post(
        f"{base.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        headers={"X-DIO-Tier": tier},
        timeout=300.0,
    )
    e2e_ms = (time.perf_counter() - t0) * 1000.0
    ttft_ms = None
    text = ""
    usage = {}
    try:
        body = r.json()
        usage = body.get("usage") or {}
        # optional engine-provided timing
        ttft_ms = body.get("dio_engine_e2e_ms")  # our HF server
        if "choices" in body and body["choices"]:
            ch = body["choices"][0]
            if "message" in ch:
                text = ch["message"].get("content") or ""
            elif "text" in ch:
                text = ch.get("text") or ""
        # some stacks put TTFT in headers
        if r.headers.get("X-TTFT-Ms"):
            ttft_ms = float(r.headers["X-TTFT-Ms"])
    except Exception:
        body = {"raw": r.text[:300]}
    return {
        "status": r.status_code,
        "e2e_ms": e2e_ms,
        "ttft_ms": ttft_ms,
        "backend": r.headers.get("X-DIO-Backend"),
        "dio_e2e_hdr": r.headers.get("X-DIO-E2E-Ms"),
        "usage": usage,
        "text_preview": (text or "")[:120],
        "ok": r.status_code == 200,
    }


def run_load(
    base: str,
    *,
    model: str,
    n: int,
    max_tokens: int,
    seed: int,
) -> Dict[str, Any]:
    random.seed(seed)
    e2e: List[float] = []
    ttft: List[float] = []
    routes: Dict[str, int] = {}
    ok = 0
    fail = 0
    samples: List[str] = []
    with httpx.Client() as client:
        # health
        try:
            h = client.get(f"{base}/healthz", timeout=5.0)
            health = h.json() if h.status_code == 200 else {"status": h.status_code}
        except Exception as e:
            return {"error": f"gateway unreachable: {e}", "ok": 0}

        for i in range(n):
            prompt = PROMPTS[i % len(PROMPTS)] + f" (seed={seed} i={i})"
            r = chat_once(client, base, prompt, model, max_tokens)
            if r["ok"]:
                ok += 1
                e2e.append(r["e2e_ms"])
                if r["ttft_ms"] is not None:
                    try:
                        ttft.append(float(r["ttft_ms"]))
                    except Exception:
                        pass
                b = r["backend"] or "?"
                routes[b] = routes.get(b, 0) + 1
                if len(samples) < 3:
                    samples.append(r["text_preview"])
            else:
                fail += 1

        metrics = {}
        try:
            metrics = client.get(f"{base}/debug/metrics", timeout=10.0).json()
        except Exception:
            pass

    return {
        "health": health,
        "n": n,
        "ok": ok,
        "fail": fail,
        "routes": routes,
        "e2e_p50_ms": pct(e2e, 50),
        "e2e_p95_ms": pct(e2e, 95),
        "e2e_p99_ms": pct(e2e, 99),
        "e2e_mean_ms": statistics.mean(e2e) if e2e else None,
        "ttft_p50_ms": pct(ttft, 50),
        "ttft_p99_ms": pct(ttft, 99),
        "ttft_n": len(ttft),
        "samples": samples,
        "prediction": (metrics.get("prediction") or {}) if metrics else {},
        "admission": (metrics.get("admission") or {}) if metrics else {},
        "workers": (metrics.get("workers") or {}) if metrics else {},
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def exp_G1_smoke(base: str, model: str, max_tokens: int) -> Dict[str, Any]:
    log("=== G1 Single-backend / smoke ===")
    out = run_load(base, model=model, n=3, max_tokens=max_tokens, seed=0)
    log(f"  smoke ok={out.get('ok')}/3 p50={out.get('e2e_p50_ms')} routes={out.get('routes')}")
    return out


def exp_G2_multiseed_matrix(
    backends: List[str],
    *,
    model: str,
    strategies: List[str],
    seeds: int,
    n_per_seed: int,
    max_tokens: int,
    dio_base_port: int,
    slo_ms: float,
    session: ClusterSession,
    tokenizer: str = "",
    admission_mode: str = "rank_only",
) -> Dict[str, Any]:
    log("=== G2 Multi-seed strategy matrix ===")
    results: Dict[str, Any] = {"strategies": {}, "seeds": seeds, "n_per_seed": n_per_seed}
    for strat in strategies:
        seed_rows = []
        for seed in range(seeds):
            port = dio_base_port + hash((strat, seed)) % 1000
            # ensure free-ish port range
            port = dio_base_port + (strategies.index(strat) * 10 + seed)
            h = start_dio(
                session,
                backends=backends,
                port=port,
                strategy=strat,
                nlms_mode="dual",
                slo_ms=slo_ms,
                admission_off=True,
                tokenizer=tokenizer,
                admission_mode=admission_mode,
                name=f"dio_{strat}_s{seed}",
            )
            url = f"http://127.0.0.1:{port}"
            if not wait_url(f"{url}/healthz", timeout=90):
                log(f"  FAIL start DIO {strat} seed={seed}")
                seed_rows.append({"seed": seed, "error": "dio_start_failed"})
                kill_proc(h.proc)
                continue
            row = run_load(url, model=model, n=n_per_seed, max_tokens=max_tokens, seed=seed)
            row["seed"] = seed
            seed_rows.append(row)
            log(f"  {strat} seed={seed}: ok={row.get('ok')} e2e_p99={row.get('e2e_p99_ms')} routes={row.get('routes')}")
            kill_proc(h.proc)
            time.sleep(0.8)

        # aggregate
        p99s = [r["e2e_p99_ms"] for r in seed_rows if r.get("e2e_p99_ms") is not None]
        p50s = [r["e2e_p50_ms"] for r in seed_rows if r.get("e2e_p50_ms") is not None]
        means = [r["e2e_mean_ms"] for r in seed_rows if r.get("e2e_mean_ms") is not None]
        oks = [r.get("ok", 0) for r in seed_rows]
        results["strategies"][strat] = {
            "per_seed": seed_rows,
            "e2e_p99": mean_std([float(x) for x in p99s]),
            "e2e_p50": mean_std([float(x) for x in p50s]),
            "e2e_mean": mean_std([float(x) for x in means]),
            "ok_sum": sum(oks),
        }
        m = results["strategies"][strat]["e2e_p99"]
        log(f"  >> {strat} p99 mean±std = {m['mean']:.1f}±{m['std']:.1f} (n={m['n']})")
    return results


def exp_G3_hetero(
    backends: List[str],
    *,
    model: str,
    seeds: int,
    n_per_seed: int,
    max_tokens: int,
    dio_port: int,
    session: ClusterSession,
    hetero_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    log("=== G3 Dual-backend heterogeneity (NLMS vs RR) ===")
    if len(backends) < 2:
        return {"skipped": True, "reason": "need ≥2 backends"}
    if hetero_meta:
        log(f"  hetero setup: {hetero_meta.get('note')}")

    out: Dict[str, Any] = {"nlms": [], "round_robin": []}
    for strat, key in [("nlms", "nlms"), ("round_robin", "round_robin")]:
        for seed in range(seeds):
            port = dio_port + (0 if strat == "nlms" else 50) + seed
            h = start_dio(
                session,
                backends=backends,
                port=port,
                strategy=strat,
                admission_off=True,
                slo_ms=180_000,
                name=f"hetero_{strat}_s{seed}",
            )
            url = f"http://127.0.0.1:{port}"
            if not wait_url(f"{url}/healthz", timeout=90):
                out[key].append({"seed": seed, "error": "start_failed"})
                kill_proc(h.proc)
                continue
            row = run_load(url, model=model, n=n_per_seed, max_tokens=max_tokens, seed=1000 + seed)
            row["seed"] = seed
            # fraction to e0 (first backend = typically fast / unthrottled)
            total = sum(row.get("routes", {}).values()) or 1
            row["frac_e0"] = row.get("routes", {}).get("e0", 0) / total
            # also capture MAPE for honesty (absolute prediction quality)
            row["mape"] = (row.get("prediction") or {}).get("mape_pct")
            out[key].append(row)
            log(
                f"  {strat} seed={seed}: frac_e0={row['frac_e0']:.2f} "
                f"p99={row.get('e2e_p99_ms')} mape={row.get('mape')} routes={row.get('routes')}"
            )
            kill_proc(h.proc)
            time.sleep(0.8)

    def agg(rows: List[dict], field: str):
        xs = [float(r[field]) for r in rows if field in r and r[field] is not None]
        return mean_std(xs)

    summary: Dict[str, Any] = {
        "setup": hetero_meta or {"throttled": False},
        "seeds": seeds,
        "n_per_seed": n_per_seed,
        "nlms_frac_e0": agg(out["nlms"], "frac_e0"),
        "rr_frac_e0": agg(out["round_robin"], "frac_e0"),
        "nlms_p99": agg(out["nlms"], "e2e_p99_ms"),
        "rr_p99": agg(out["round_robin"], "e2e_p99_ms"),
        "nlms_mape": agg(out["nlms"], "mape"),
        "rr_mape": agg(out["round_robin"], "mape"),
        "per_seed": out,
    }
    # improvement
    imps = []
    for i in range(min(len(out["nlms"]), len(out["round_robin"]))):
        a, b = out["nlms"][i], out["round_robin"][i]
        if a.get("e2e_p99_ms") and b.get("e2e_p99_ms") and b["e2e_p99_ms"] > 0:
            imps.append((b["e2e_p99_ms"] - a["e2e_p99_ms"]) / b["e2e_p99_ms"] * 100.0)
    summary["p99_improvement_pct"] = mean_std(imps)
    log(
        f"  >> NLMS frac_e0={summary['nlms_frac_e0']['mean']:.3f}±{summary['nlms_frac_e0']['std']:.3f} "
        f"p99_impr%={summary['p99_improvement_pct']['mean']:.1f}±{summary['p99_improvement_pct']['std']:.1f} "
        f"MAPE={summary['nlms_mape']['mean']:.1f}%±{summary['nlms_mape']['std']:.1f}% "
        f"(n={seeds} seeds, throttled={bool((hetero_meta or {}).get('throttled'))})"
    )
    return summary


def exp_G4_dual_vs_single(
    backends: List[str],
    *,
    model: str,
    seeds: int,
    n_per_seed: int,
    max_tokens: int,
    dio_port: int,
    session: ClusterSession,
) -> Dict[str, Any]:
    log("=== G4 Dual vs SINGLE NLMS ===")
    out: Dict[str, Any] = {"dual": [], "single": []}
    for mode, key in [("dual", "dual"), ("single", "single")]:
        for seed in range(seeds):
            port = dio_port + (0 if mode == "dual" else 30) + seed
            h = start_dio(
                session,
                backends=backends,
                port=port,
                strategy="nlms",
                nlms_mode=mode,
                admission_off=True,
                slo_ms=180_000,
                name=f"mode_{mode}_s{seed}",
            )
            url = f"http://127.0.0.1:{port}"
            if not wait_url(f"{url}/healthz", timeout=90):
                out[key].append({"seed": seed, "error": "start_failed"})
                kill_proc(h.proc)
                continue
            row = run_load(url, model=model, n=n_per_seed, max_tokens=max_tokens, seed=2000 + seed)
            row["seed"] = seed
            row["mape"] = (row.get("prediction") or {}).get("mape_pct")
            out[key].append(row)
            log(f"  {mode} seed={seed}: p99={row.get('e2e_p99_ms')} mape={row.get('mape')}")
            kill_proc(h.proc)
            time.sleep(0.8)

    def agg(rows, field):
        xs = [float(r[field]) for r in rows if r.get(field) is not None]
        return mean_std(xs)

    return {
        "dual_p99": agg(out["dual"], "e2e_p99_ms"),
        "single_p99": agg(out["single"], "e2e_p99_ms"),
        "dual_mape": agg(out["dual"], "mape"),
        "single_mape": agg(out["single"], "mape"),
        "per_seed": out,
    }


def exp_G5_admission(
    backends: List[str],
    *,
    model: str,
    n: int,
    max_tokens: int,
    dio_port: int,
    tight_slo_ms: float,
    session: ClusterSession,
) -> Dict[str, Any]:
    log("=== G5 Admission ON vs OFF ===")
    out = {}
    for admit_off, tag in [(True, "off"), (False, "on")]:
        h = start_dio(
            session,
            backends=backends,
            port=dio_port + (0 if admit_off else 1),
            strategy="nlms",
            admission_off=admit_off,
            slo_ms=tight_slo_ms if not admit_off else 180_000,
            name=f"admit_{tag}",
        )
        url = f"http://127.0.0.1:{dio_port + (0 if admit_off else 1)}"
        if not wait_url(f"{url}/healthz", timeout=90):
            out[tag] = {"error": "start_failed"}
            kill_proc(h.proc)
            continue
        # longer prompts to stress e2e prediction
        rows = []
        with httpx.Client() as client:
            for i in range(n):
                prompt = ("Write a detailed paragraph about space. " * 3) + f" #{i}"
                r = chat_once(client, url, prompt, model, max_tokens=max_tokens)
                rows.append(r)
        ok = sum(1 for r in rows if r["ok"])
        rej = sum(1 for r in rows if r["status"] == 503)
        e2e = [r["e2e_ms"] for r in rows if r["ok"]]
        try:
            adm = httpx.get(f"{url}/debug/admission", timeout=10).json()
        except Exception:
            adm = {}
        out[tag] = {
            "ok": ok,
            "http_503": rej,
            "e2e_p99_ms": pct(e2e, 99),
            "admission": adm,
            "slo_ms": tight_slo_ms if not admit_off else 180_000,
        }
        log(f"  admission_{tag}: ok={ok} 503={rej} p99={out[tag]['e2e_p99_ms']}")
        kill_proc(h.proc)
        time.sleep(0.8)
    return out


def exp_G7_cpu_t2() -> Dict[str, Any]:
    """Always-safe multi-seed microbench (no GPU)."""
    log("=== G7 CPU multi-seed T2 microbench (library) ===")
    try:
        from run_t2_multiseed import main as t2_main
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            t2_main()
        path = ROOT / "results_validation" / "t2_multiseed.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"raw": buf.getvalue()[-500:]}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_tables(summary: Dict[str, Any], out: Path) -> None:
    rows = []
    g2 = summary.get("G2_matrix") or {}
    for strat, block in (g2.get("strategies") or {}).items():
        p99 = block.get("e2e_p99") or {}
        p50 = block.get("e2e_p50") or {}
        rows.append(
            {
                "suite": "G2",
                "strategy": strat,
                "e2e_p99_mean": p99.get("mean"),
                "e2e_p99_std": p99.get("std"),
                "e2e_p50_mean": p50.get("mean"),
                "e2e_p50_std": p50.get("std"),
                "ok_sum": block.get("ok_sum"),
            }
        )
    g3 = summary.get("G3_hetero") or {}
    if g3 and not g3.get("skipped"):
        rows.append(
            {
                "suite": "G3",
                "strategy": "nlms_vs_rr",
                "nlms_frac_e0_mean": (g3.get("nlms_frac_e0") or {}).get("mean"),
                "nlms_frac_e0_std": (g3.get("nlms_frac_e0") or {}).get("std"),
                "p99_impr_mean": (g3.get("p99_improvement_pct") or {}).get("mean"),
                "p99_impr_std": (g3.get("p99_improvement_pct") or {}).get("std"),
            }
        )
    csv_path = out / "tables.csv"
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        log(f"Wrote {csv_path}")

    # paper snippets
    lines = [
        "# Paper-ready snippets (from GPU cluster validation)",
        f"Generated: {summary.get('generated_at')}",
        f"Model: {summary.get('args', {}).get('model')}",
        f"Engine mode: {summary.get('args', {}).get('engine_mode')}",
        "",
    ]
    if g2.get("strategies"):
        lines.append("## G2 multi-seed e2e p99 (mean±std)")
        for strat, block in g2["strategies"].items():
            p = block.get("e2e_p99") or {}
            if p.get("n"):
                lines.append(
                    f"- **{strat}**: ${p['mean']:.1f} \\pm {p['std']:.1f}$ ms "
                    f"(n={p['n']} seeds)"
                )
        lines.append("")
    if g3 and not g3.get("skipped"):
        setup = g3.get("setup") or {}
        lines.append("## G3 heterogeneity (real engines)")
        lines.append(
            f"- setup: throttled={setup.get('throttled')} mult={setup.get('slow_mult')} "
            f"seeds={g3.get('seeds')} note={setup.get('note')}"
        )
        f = g3.get("nlms_frac_e0") or {}
        imp = g3.get("p99_improvement_pct") or {}
        mape = g3.get("nlms_mape") or {}
        lines.append(
            f"- NLMS fraction to first (fast) backend: "
            f"${f.get('mean', float('nan')):.3f} \\pm {f.get('std', float('nan')):.3f}$"
        )
        lines.append(
            f"- p99 improvement vs RR: "
            f"${imp.get('mean', float('nan')):.1f}\\% \\pm {imp.get('std', float('nan')):.1f}\\%$"
        )
        if mape.get("n"):
            lines.append(
                f"- NLMS absolute MAPE (honest): "
                f"${mape['mean']:.1f}\\% \\pm {mape['std']:.1f}\\%$ "
                f"(high MAPE does not preclude useful ranking)"
            )
        lines.append("")
    g4 = summary.get("G4_dual_single") or {}
    if g4:
        lines.append("## G4 dual vs single (MAPE)")
        for k in ("dual_p99", "single_p99", "dual_mape", "single_mape"):
            p = g4.get(k) or {}
            if p.get("n"):
                lines.append(f"- {k}: ${p['mean']:.2f} \\pm {p['std']:.2f}$")
        lines.append(
            "- Note: report MAPE explicitly; DIO's routing claim is relative ranking, "
            "not point-accurate ms prediction."
        )
    snip = out / "paper_snippets.md"
    snip.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote {snip}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DIO grand GPU-cluster validation (real engines + multi-seed)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--engine-mode", choices=["vllm", "hf", "external"], default="hf")
    p.add_argument(
        "--backends",
        type=str,
        default="",
        help="Comma-separated base URLs (required for external; optional override)",
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU indices")
    p.add_argument("--seeds", type=int, default=3, help="≥3 recommended for paper")
    p.add_argument("--requests-per-seed", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--engine-base-port", type=int, default=18000)
    p.add_argument("--dio-base-port", type=int, default=19000)
    p.add_argument("--slo-ms", type=float, default=120000.0, help="Loose SLO for routing A/B")
    p.add_argument("--tight-slo-ms", type=float, default=3000.0, help="For admission test")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument(
        "--strategies",
        type=str,
        default="nlms,rls,round_robin,least_loaded",
        help="Comma list for G2 matrix (include rls for head-to-head)",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="HF tokenizer name for NLMS token feature (e.g. Qwen/Qwen2.5-3B-Instruct)",
    )
    p.add_argument(
        "--admission-mode",
        type=str,
        default="rank_only",
        help="absolute|empirical|rank_only (routing A/B uses rank_only by default)",
    )
    p.add_argument("--skip-g2", action="store_true")
    p.add_argument("--skip-g3", action="store_true")
    p.add_argument("--skip-g4", action="store_true")
    p.add_argument("--skip-g5", action="store_true")
    p.add_argument("--skip-g7", action="store_true")
    p.add_argument(
        "--hetero-slow-mult",
        type=float,
        default=2.0,
        help=(
            "For G3: multiply second backend e2e via delay proxy (vLLM/external) "
            "or HF --latency-mult. Use 1.0 for homogeneous G3. Default 2.0."
        ),
    )
    p.add_argument(
        "--hetero-seeds",
        type=int,
        default=0,
        help="Seeds for G3 only (0 = use --seeds). Prefer 5 for real-GPU hetero claims.",
    )
    p.add_argument(
        "--proxy-port",
        type=int,
        default=18101,
        help="Port for latency delay proxy wrapping the slow peer",
    )
    p.add_argument("--quick", action="store_true", help="Fewer seeds/requests")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.seeds = min(args.seeds, 2)
        args.requests_per_seed = min(args.requests_per_seed, 12)
        if args.hetero_seeds:
            args.hetero_seeds = min(args.hetero_seeds, 2)
    hetero_seeds = args.hetero_seeds if args.hetero_seeds and args.hetero_seeds > 0 else args.seeds

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = ClusterSession(out)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "run_gpu_cluster_validation.py",
        "args": vars(args),
        "hetero_seeds_effective": hetero_seeds,
        "env": probe_env(),
        "purpose": [
            "Multi-seed real-engine matrix (homogeneous dual-GPU)",
            "Real-GPU heterogeneity via delay-proxy throttle (G3) with mean±std",
            "Dual vs single NLMS + MAPE honesty",
            "Admission under load",
            "CPU slope-skew microbench (G7) for ranking evidence",
        ],
    }

    log("=" * 64)
    log("DIO GPU CLUSTER VALIDATION")
    log("=" * 64)
    log(f"dio={summary['env'].get('dio_version')} torch={summary['env'].get('torch')}")
    log(f"nvidia-smi: {summary['env'].get('nvidia_smi')}")

    if summary["env"].get("dio_import_error"):
        log("ERROR: install package first: cd dio-serve && pip install -e .")
        return 2

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    backends: List[str] = []

    try:
        # ----- Start or attach engines -----
        if args.engine_mode == "external" or args.backends.strip():
            backends = [b.strip() for b in args.backends.split(",") if b.strip()]
            if not backends:
                log("ERROR: --engine-mode external requires --backends")
                return 2
            log(f"Using external backends: {backends}")
            for b in backends:
                if not wait_url(b.rstrip("/") + "/v1/models", timeout=30) and not wait_url(
                    b.rstrip("/") + "/health", timeout=5
                ):
                    log(f"WARN: backend not responding yet: {b}")
        elif args.engine_mode == "vllm":
            if len(gpus) < 1:
                log("ERROR: need --gpus for vllm mode")
                return 2
            for i, gpu in enumerate(gpus):
                port = args.engine_base_port + i
                start_vllm_engine(
                    session,
                    gpu=gpu,
                    port=port,
                    model=args.model,
                    max_model_len=args.max_model_len,
                    gpu_mem_util=args.gpu_mem_util,
                    name=f"vllm_gpu{gpu}",
                )
                backends.append(f"http://127.0.0.1:{port}")
            for b in backends:
                log(f"Waiting for {b} (vLLM can take several minutes)...")
                if not wait_url(b + "/v1/models", timeout=900):
                    log(f"ERROR: engine failed: {b} — see logs/")
                    # print last log lines
                    for h in session.handles:
                        if h.log_path.exists():
                            log(f"--- {h.log_path.name} tail ---")
                            log("\n".join(h.log_path.read_text(errors="replace").splitlines()[-30:]))
                    return 1
        else:  # hf
            if not ENGINE_HF.exists():
                log(f"ERROR: missing {ENGINE_HF}")
                return 2
            # engine 0
            g0 = gpus[0] if gpus else "0"
            start_hf_engine(
                session,
                gpu=g0,
                port=args.engine_base_port,
                model=args.model,
                latency_mult=1.0,
                name="hf_fast",
            )
            backends.append(f"http://127.0.0.1:{args.engine_base_port}")
            if len(gpus) >= 2:
                # Keep second HF engine at mult=1.0; G3 applies delay proxy so
                # both vLLM and HF paths share the same throttle mechanism.
                start_hf_engine(
                    session,
                    gpu=gpus[1],
                    port=args.engine_base_port + 1,
                    model=args.model,
                    latency_mult=1.0,
                    name="hf_gpu1",
                )
                backends.append(f"http://127.0.0.1:{args.engine_base_port + 1}")
            else:
                log(
                    "NOTE: single GPU — only one HF engine started "
                    "(G3 hetero needs --gpus 0,1 or --backends a,b). "
                    "Do not co-locate two full models on one 6–8GB card."
                )

            for b in backends:
                log(f"Waiting for {b}...")
                if not wait_url(b + "/health", timeout=900):
                    log(f"ERROR: HF engine failed: {b}")
                    for h in session.handles:
                        if h.log_path.exists():
                            log("\n".join(h.log_path.read_text(errors="replace").splitlines()[-40:]))
                    return 1

        summary["backends"] = backends

        # ----- G1 smoke -----
        # temporary dio for smoke
        smoke_port = args.dio_base_port
        h = start_dio(
            session,
            backends=backends[:1],
            port=smoke_port,
            strategy="nlms",
            admission_off=True,
            slo_ms=args.slo_ms,
            name="dio_smoke",
        )
        smoke_url = f"http://127.0.0.1:{smoke_port}"
        if not wait_url(f"{smoke_url}/healthz", timeout=90):
            log("ERROR: DIO smoke failed to start")
            return 1
        summary["G1_smoke"] = exp_G1_smoke(smoke_url, args.model, args.max_tokens)
        if summary["G1_smoke"].get("ok", 0) < 1:
            log("ERROR: smoke produced no successful requests — aborting grand suite")
            return 1
        kill_proc(h.proc)
        time.sleep(1)

        # ----- G2 multi-seed matrix -----
        if not args.skip_g2:
            summary["G2_matrix"] = exp_G2_multiseed_matrix(
                backends,
                model=args.model,
                strategies=strategies,
                seeds=args.seeds,
                n_per_seed=args.requests_per_seed,
                max_tokens=args.max_tokens,
                dio_base_port=args.dio_base_port + 100,
                slo_ms=args.slo_ms,
                session=session,
                tokenizer=args.tokenizer,
                admission_mode=args.admission_mode,
            )

        # ----- G3 hetero (optionally throttle second peer) -----
        # Homogeneous G2 uses raw backends; G3 may wrap e1 with delay proxy.
        if not args.skip_g3 and len(backends) >= 2:
            hetero_backends, hetero_meta = make_hetero_backends(
                session,
                backends,
                slow_mult=args.hetero_slow_mult,
                proxy_port=args.proxy_port,
            )
            summary["G3_hetero"] = exp_G3_hetero(
                hetero_backends,
                model=args.model,
                seeds=hetero_seeds,
                n_per_seed=args.requests_per_seed,
                max_tokens=args.max_tokens,
                dio_port=args.dio_base_port + 200,
                session=session,
                hetero_meta=hetero_meta,
            )
        elif not args.skip_g3:
            summary["G3_hetero"] = {"skipped": True, "reason": "need 2 backends"}

        # ----- G4 dual/single -----
        if not args.skip_g4:
            summary["G4_dual_single"] = exp_G4_dual_vs_single(
                backends,
                model=args.model,
                seeds=args.seeds,
                n_per_seed=max(10, args.requests_per_seed // 2),
                max_tokens=args.max_tokens,
                dio_port=args.dio_base_port + 300,
                session=session,
            )

        # ----- G5 admission -----
        if not args.skip_g5:
            summary["G5_admission"] = exp_G5_admission(
                backends,
                model=args.model,
                n=max(12, args.requests_per_seed),
                max_tokens=args.max_tokens,
                dio_port=args.dio_base_port + 400,
                tight_slo_ms=args.tight_slo_ms,
                session=session,
            )

        # ----- G7 CPU T2 -----
        if not args.skip_g7:
            summary["G7_t2_cpu_multiseed"] = exp_G7_cpu_t2()

        summary["status"] = "ok"

    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        log("Interrupted")
    except Exception as e:
        summary["status"] = "error"
        summary["error"] = str(e)
        log(f"FATAL: {e}")
        import traceback

        traceback.print_exc()
    finally:
        session.cleanup()

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_tables(summary, out)

    print("\n" + "=" * 64)
    print("GPU CLUSTER VALIDATION COMPLETE")
    print("=" * 64)
    print(f"Status: {summary.get('status')}")
    print(f"Backends: {summary.get('backends')}")
    g2 = summary.get("G2_matrix") or {}
    for strat, block in (g2.get("strategies") or {}).items():
        p = block.get("e2e_p99") or {}
        if p.get("n"):
            print(f"  {strat}: p99 {p['mean']:.1f}±{p['std']:.1f} ms (n={p['n']})")
    g3 = summary.get("G3_hetero") or {}
    if g3 and not g3.get("skipped"):
        imp = g3.get("p99_improvement_pct") or {}
        fr = g3.get("nlms_frac_e0") or {}
        mp = g3.get("nlms_mape") or {}
        setup = g3.get("setup") or {}
        print(
            f"  hetero (throttled={setup.get('throttled')}, n={g3.get('seeds')}): "
            f"frac_fast={fr.get('mean')}±{fr.get('std')} "
            f"p99_impr={imp.get('mean')}±{imp.get('std')}% "
            f"MAPE={mp.get('mean')}±{mp.get('std')}%"
        )
    print(f"\nResults → {out}")
    print("  summary.json  tables.csv  paper_snippets.md  logs/")
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
