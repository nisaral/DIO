#!/usr/bin/env python3
"""
DIO Camera-Ready Experiment Suite
=================================
One script for GPU-cluster / multi-T4 / mock-only novelty validation.

Implements paper novelty experiments:
  N1 Dual-timescale NLMS  — DUAL vs SINGLE under burst jitter + thermal drift
  N2 Admission goodput    — admission ON vs OFF under overload (reject if min S_w > SLO)
  N3 Joint cost / multi-tier — small+large workers, tier-aware routing
  N4 Ablations            — full | no_queue | no_vram | no_tier | no_cache | no_dual | STATIC
  N5 Control-plane scale  — many mock workers, scheduling stays light
  E1 Main e2e matrix      — NLMS / RR / LL / RLS / STATIC × workloads (real or mock)

Usage (from repo DIO/ directory):
  # Mock-only (no GPU, full novelty suite, ~15–40 min)
  python benchmarks/camera_ready_suite.py --mode mock --quick

  # 2× T4 (or any multi-GPU): real workers pinned to devices
  python benchmarks/camera_ready_suite.py --mode real --gpus 0,1 --model meta-llama/Llama-3.2-3B-Instruct

  # Novelty only (skip long e2e)
  python benchmarks/camera_ready_suite.py --mode mock --only novelty

  # Full paper suite
  python benchmarks/camera_ready_suite.py --mode real --gpus 0,1 --seeds 3

Env knobs (also set by this script per experiment):
  SCHEDULER_STRATEGY, NLMS_MODE, DIO_ABLATION, DIO_SLO_MS, DIO_ADMISSION_OFF,
  STATIC_SLOPE, STATIC_INTERCEPT, MODEL_ID, WORKLOAD_FILE

Outputs:
  benchmarks/results_camera_ready/
    summary.json          — all experiment metrics
    predictions_*.json    — MAPE traces for dual vs single
    figures/              — auto plots when matplotlib available
    logs/                 — manager/worker logs
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCH_DIR = Path(__file__).resolve().parent
DIO_ROOT = BENCH_DIR.parent
RESULTS = BENCH_DIR / "results_camera_ready"
LOGS = RESULTS / "logs"
FIGS = RESULTS / "figures"
DATA = BENCH_DIR / "data"

MANAGER_HTTP = os.environ.get("DIO_HTTP", "http://127.0.0.1:8085")
MANAGER_GRPC = os.environ.get("DIO_GRPC", "127.0.0.1:50055")
WORKER_SCRIPT = BENCH_DIR / "worker_gpu.py"
LOCUST_FILE = BENCH_DIR / "real_world" / "locustfile.py"

# Processes we own (for cleanup)
_PROCS: List[subprocess.Popen] = []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------
def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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


def cleanup_all() -> None:
    log("Cleanup: stopping managed processes...")
    for p in list(_PROCS):
        kill_tree(p)
    _PROCS.clear()
    # Best-effort orphan sweep (Linux clusters)
    if os.name != "nt":
        for pat in ("dio-manager", "worker_gpu.py", "cmd/manager", "locust"):
            subprocess.run(["pkill", "-9", "-f", pat],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_json(url: str, method: str = "GET", body: Optional[dict] = None, timeout: float = 30.0) -> Tuple[int, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw) if raw else {"error": str(e)}
        except json.JSONDecodeError:
            return e.code, {"error": raw}
    except Exception as e:
        return 0, {"error": str(e)}


def wait_http(path: str = "/healthz", timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        code, _ = http_json(f"{MANAGER_HTTP}{path}", timeout=2.0)
        if code == 200:
            return True
        time.sleep(0.4)
    return False


def build_manager(force: bool = False) -> Path:
    out = DIO_ROOT / ("dio-manager.exe" if os.name == "nt" else "dio-manager")
    if out.exists() and not force:
        # Rebuild if source newer
        try:
            newest = max((p.stat().st_mtime for p in (DIO_ROOT / "internal" / "scheduler").rglob("*.go")), default=0)
            if out.stat().st_mtime >= newest:
                return out
        except Exception:
            pass
    log(f"Building manager → {out}")
    r = subprocess.run(
        ["go", "build", "-o", str(out), "./cmd/manager"],
        cwd=str(DIO_ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        log(r.stderr)
        raise RuntimeError("go build failed")
    return out


# ---------------------------------------------------------------------------
# Start manager / workers
# ---------------------------------------------------------------------------
def start_manager(
    strategy: str = "NLMS",
    nlms_mode: str = "DUAL",
    ablation: str = "full",
    slo_ms: float = 5000.0,
    admission_off: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    binary = build_manager()
    env = os.environ.copy()
    env["SCHEDULER_STRATEGY"] = strategy
    env["NLMS_MODE"] = nlms_mode
    env["DIO_ABLATION"] = ablation
    env["DIO_SLO_MS"] = str(slo_ms)
    env["DIO_ADMISSION_OFF"] = "1" if admission_off else "0"
    env["TELEMETRY_FILE"] = str(LOGS / f"telemetry_{strategy}_{ablation}.csv")
    if extra_env:
        env.update(extra_env)
    log_path = LOGS / f"manager_{strategy}_{nlms_mode}_{ablation}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    kwargs: Dict[str, Any] = {"cwd": str(DIO_ROOT), "env": env, "stdout": log_f, "stderr": subprocess.STDOUT}
    if os.name != "nt":
        kwargs["preexec_fn"] = os.setsid
    proc = subprocess.Popen([str(binary)], **kwargs)
    _PROCS.append(proc)
    if not wait_http("/healthz", timeout=45):
        raise RuntimeError(f"Manager failed to start; see {log_path}")
    log(f"Manager up strategy={strategy} mode={nlms_mode} ablation={ablation} slo={slo_ms} admit_off={admission_off}")
    return proc


def start_worker(
    worker_id: str,
    port: int,
    mock: bool,
    model_id: str,
    cuda_device: Optional[str] = None,
    tier: str = "small",
    vram_mb: int = 15000,
    latency_profile: Optional[str] = None,
    profile_role: Optional[str] = None,
    latency_mult: float = 1.0,
    mock_seed: int = 42,
) -> subprocess.Popen:
    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    cmd = [
        sys.executable, str(WORKER_SCRIPT),
        "--worker-id", worker_id,
        "--port", str(port),
        "--model-id", model_id,
        "--manager-addr", MANAGER_GRPC,
        "--vram", str(vram_mb),
        "--latency-mult", str(latency_mult),
        "--tier", tier,
    ]
    if mock_seed is not None:
        cmd.extend(["--mock-seed", str(mock_seed)])
    if mock:
        cmd.append("--mock")
    if latency_profile:
        cmd.extend(["--latency-profile", latency_profile])
    if profile_role:
        cmd.extend(["--profile-role", profile_role])
    env["WORKER_TIER"] = tier
    env["WORKER_HOST"] = "127.0.0.1"

    log_path = LOGS / f"worker_{worker_id}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    kwargs: Dict[str, Any] = {"cwd": str(DIO_ROOT), "env": env, "stdout": log_f, "stderr": subprocess.STDOUT}
    if os.name != "nt":
        kwargs["preexec_fn"] = os.setsid
    proc = subprocess.Popen(cmd, **kwargs)
    _PROCS.append(proc)
    return proc


def wait_workers(n: int, timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        code, body = http_json(f"{MANAGER_HTTP}/debug/workers", timeout=3.0)
        if code == 200 and isinstance(body, dict) and body.get("worker_count", 0) >= n:
            log(f"All {n} workers registered")
            return True
        time.sleep(1.0)
    return False


def reset_stats() -> None:
    http_json(f"{MANAGER_HTTP}/debug/reset_stats", method="POST", body={}, timeout=5.0)


def fetch_admission() -> dict:
    code, body = http_json(f"{MANAGER_HTTP}/debug/admission", timeout=5.0)
    return body if code == 200 and isinstance(body, dict) else {}


def fetch_predictions(limit: int = 5000) -> dict:
    code, body = http_json(f"{MANAGER_HTTP}/debug/predictions?limit={limit}", timeout=10.0)
    return body if code == 200 and isinstance(body, dict) else {}


def fetch_metrics() -> dict:
    code, body = http_json(f"{MANAGER_HTTP}/debug/metrics", timeout=10.0)
    return body if code == 200 and isinstance(body, dict) else {}


# ---------------------------------------------------------------------------
# Built-in load generator (no Locust required for novelty tests)
# ---------------------------------------------------------------------------
@dataclass
class LoadResult:
    name: str
    n_ok: int = 0
    n_fail: int = 0
    n_503: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    wall_s: float = 0.0
    admission: dict = field(default_factory=dict)
    prediction: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        lats = sorted(self.latencies_ms)
        def pct(p):
            if not lats:
                return None
            i = min(len(lats) - 1, int(round((p / 100.0) * (len(lats) - 1))))
            return lats[i]
        goodput = self.n_ok / self.wall_s if self.wall_s > 0 else 0.0
        # Venue-style goodput under SLO: completions under SLO / wall
        under = float(self.admission.get("completed_under_slo", 0) or 0)
        goodput_slo = under / self.wall_s if self.wall_s > 0 else 0.0
        return {
            "name": self.name,
            "n_ok": self.n_ok,
            "n_fail": self.n_fail,
            "n_503": self.n_503,
            "rps": goodput,
            "goodput_under_slo_rps": goodput_slo,
            "p50_ms": pct(50),
            "p95_ms": pct(95),
            "p99_ms": pct(99),
            "mean_ms": statistics.mean(lats) if lats else None,
            "wall_s": self.wall_s,
            "admission": self.admission,
            "prediction_summary": {
                "count": self.prediction.get("count"),
                "mae_ms": self.prediction.get("mae_ms"),
                "mape_pct": self.prediction.get("mape_pct"),
            },
            "extra": self.extra,
        }


def make_prompts(n: int = 200, long_frac: float = 0.2) -> List[dict]:
    """Synthetic prompts if trace files missing; mixes short/long for tier tests."""
    prompts = []
    for i in range(n):
        if random.random() < long_frac:
            text = ("Summarize this document. " + ("lorem ipsum " * 200))[:1600]
            tier = "large"
        else:
            text = f"Hello, quick question number {i}: what is 2+2?"
            tier = "small"
        prompts.append({"prompt": text, "tier": tier})
    # Prefer real traces if present
    for name in ("sharegpt.jsonl", "arxiv.jsonl", "azure_code.jsonl"):
        path = DATA / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        prompt = d.get("prompt")
                        if not prompt and "conversations" in d:
                            prompt = d["conversations"][0].get("value", "")
                        if prompt:
                            prompts.append({
                                "prompt": prompt[:2000],
                                "tier": d.get("tier", "small"),
                            })
                    except Exception:
                        pass
            break
    random.shuffle(prompts)
    return prompts or [{"prompt": "hi", "tier": "small"}]


def run_load(
    name: str,
    duration_s: float,
    concurrency: int,
    prompts: List[dict],
    open_loop_rps: Optional[float] = None,
    max_requests: Optional[int] = None,
) -> LoadResult:
    """Closed-loop (concurrency workers) or open-loop (target RPS) HTTP load."""
    reset_stats()
    result = LoadResult(name=name)
    stop = threading.Event()
    lock = threading.Lock()
    idx = [0]
    t0 = time.time()

    def one_request() -> None:
        with lock:
            item = prompts[idx[0] % len(prompts)]
            idx[0] += 1
        payload = {
            "model_id": os.environ.get("MODEL_ID", "mock"),
            "prompt": item["prompt"],
            "tier": item.get("tier", "small"),
        }
        start = time.perf_counter()
        code, body = http_json(f"{MANAGER_HTTP}/api/generate", method="POST", body=payload, timeout=180.0)
        lat = (time.perf_counter() - start) * 1000.0
        with lock:
            if code == 200:
                result.n_ok += 1
                result.latencies_ms.append(lat)
            elif code == 503:
                result.n_503 += 1
                result.n_fail += 1
            else:
                result.n_fail += 1

    def closed_loop_worker() -> None:
        while not stop.is_set():
            if max_requests is not None:
                with lock:
                    done = result.n_ok + result.n_fail
                if done >= max_requests:
                    break
            one_request()

    def open_loop() -> None:
        assert open_loop_rps and open_loop_rps > 0
        interval = 1.0 / open_loop_rps
        next_t = time.time()
        while not stop.is_set():
            if max_requests is not None and (result.n_ok + result.n_fail) >= max_requests:
                break
            threading.Thread(target=one_request, daemon=True).start()
            next_t += interval
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)

    if open_loop_rps:
        thr = threading.Thread(target=open_loop, daemon=True)
        thr.start()
        thr_list = [thr]
    else:
        thr_list = []
        for _ in range(max(1, concurrency)):
            t = threading.Thread(target=closed_loop_worker, daemon=True)
            t.start()
            thr_list.append(t)

    time.sleep(duration_s)
    stop.set()
    time.sleep(0.5)
    result.wall_s = time.time() - t0
    result.admission = fetch_admission()
    result.prediction = fetch_predictions(5000)
    return result


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def exp_dual_vs_single(args, prompts: List[dict]) -> List[dict]:
    """N1: Dual-timescale vs single-µ under burst jitter + thermal drift (mock)."""
    log("=== N1 Dual-timescale vs SINGLE (burst + thermal) ===")
    results = []
    for mode, profile_fast, profile_slow in [
        ("DUAL", "burst_jitter_fast", "thermal_drift_slow"),
        ("SINGLE", "burst_jitter_fast", "thermal_drift_slow"),
    ]:
        cleanup_all()
        start_manager(strategy="NLMS", nlms_mode=mode, ablation="full" if mode == "DUAL" else "no_dual",
                      slo_ms=args.slo_ms, admission_off=True)
        start_worker("fast", 50060, mock=True, model_id=args.model, tier="small",
                     latency_profile=profile_fast, vram_mb=20000, mock_seed=1)
        start_worker("slow", 50061, mock=True, model_id=args.model, tier="small",
                     latency_profile=profile_slow, vram_mb=20000, mock_seed=2)
        if not wait_workers(2, 60):
            raise RuntimeError("workers not ready for N1")
        time.sleep(1)
        lr = run_load(f"n1_{mode.lower()}", duration_s=args.novelty_duration, concurrency=8, prompts=prompts)
        # Save full prediction trace
        pred_path = RESULTS / f"predictions_n1_{mode.lower()}.json"
        pred_path.write_text(json.dumps(lr.prediction, indent=2), encoding="utf-8")
        d = lr.to_dict()
        d["nlms_mode"] = mode
        results.append(d)
        log(f"  {mode}: p99={d['p99_ms']} mape={d['prediction_summary'].get('mape_pct')}")
    cleanup_all()
    return results


def exp_admission_goodput(args, prompts: List[dict]) -> List[dict]:
    """N2: Admission ON vs OFF under overload — goodput should improve with admission."""
    log("=== N2 Admission as goodput optimizer ===")
    results = []
    # Tight SLO relative to mock latency so overload triggers rejects
    slo = min(args.slo_ms, 800.0)
    for admit_off, tag in [(True, "admission_off"), (False, "admission_on")]:
        cleanup_all()
        start_manager(strategy="NLMS", nlms_mode="DUAL", ablation="full",
                      slo_ms=slo, admission_off=admit_off)
        # Slow mocks so capacity is low
        start_worker("w0", 50060, mock=True, model_id=args.model, latency_profile="thermal_drift_slow",
                     latency_mult=1.5, vram_mb=8000)
        start_worker("w1", 50061, mock=True, model_id=args.model, latency_profile="thermal_drift_slow",
                     latency_mult=1.5, vram_mb=8000)
        if not wait_workers(2, 60):
            raise RuntimeError("workers not ready for N2")
        # Open-loop flood
        lr = run_load(
            f"n2_{tag}",
            duration_s=args.novelty_duration,
            concurrency=1,
            prompts=prompts,
            open_loop_rps=args.overload_rps,
        )
        d = lr.to_dict()
        d["admission_off"] = admit_off
        d["slo_ms"] = slo
        results.append(d)
        log(f"  {tag}: ok={d['n_ok']} 503={d['n_503']} goodput_slo_rps={d['goodput_under_slo_rps']:.3f} "
            f"p99={d['p99_ms']}")
    cleanup_all()
    return results


def exp_multi_tier(args, prompts: List[dict]) -> List[dict]:
    """N3: Joint tier + latency + VRAM cost on multi-model tiers."""
    log("=== N3 Multi-tier joint cost ===")
    results = []
    for ablation, tag in [("full", "tier_on"), ("no_tier", "tier_off")]:
        cleanup_all()
        start_manager(strategy="NLMS", nlms_mode="DUAL", ablation=ablation,
                      slo_ms=args.slo_ms, admission_off=True)
        start_worker("small0", 50060, mock=True, model_id=args.model, tier="small",
                     latency_profile="tier_small_fast", vram_mb=8000)
        start_worker("large0", 50061, mock=True, model_id=args.model, tier="large",
                     latency_profile="tier_large_slow", vram_mb=20000)
        if not wait_workers(2, 60):
            raise RuntimeError("workers not ready for N3")
        # Force mixed tiers
        mixed = []
        for i, p in enumerate(prompts):
            q = dict(p)
            q["tier"] = "large" if i % 3 == 0 else "small"
            mixed.append(q)
        lr = run_load(f"n3_{tag}", duration_s=args.novelty_duration, concurrency=6, prompts=mixed)
        d = lr.to_dict()
        # Routing distribution from decision log
        metrics = fetch_metrics()
        decisions = metrics.get("decisions") or []
        counts: Dict[str, int] = {}
        for dec in decisions:
            wid = dec.get("worker_id", "?")
            counts[wid] = counts.get(wid, 0) + 1
        d["routing_counts"] = counts
        d["ablation"] = ablation
        results.append(d)
        log(f"  {tag}: routing={counts} p99={d['p99_ms']}")
    cleanup_all()
    return results


def exp_ablations(args, prompts: List[dict]) -> List[dict]:
    """N4: Component ablations on joint cost function."""
    log("=== N4 Ablations ===")
    results = []
    variants = [
        ("NLMS", "DUAL", "full"),
        ("NLMS", "DUAL", "no_queue"),
        ("NLMS", "DUAL", "no_vram"),
        ("NLMS", "DUAL", "no_tier"),
        ("NLMS", "DUAL", "no_cache"),
        ("NLMS", "SINGLE", "no_dual"),
        ("STATIC", "DUAL", "full"),
        ("RoundRobin", "DUAL", "full"),
        ("LeastLoaded", "DUAL", "full"),
        ("RLS", "DUAL", "full"),
    ]
    if args.quick:
        variants = variants[:5] + [variants[-3]]  # shorter

    for strategy, mode, abl in variants:
        cleanup_all()
        extra = {}
        if strategy == "STATIC":
            extra["STATIC_SLOPE"] = "1.2"
            extra["STATIC_INTERCEPT"] = "80"
        start_manager(strategy=strategy, nlms_mode=mode, ablation=abl,
                      slo_ms=args.slo_ms, admission_off=True, extra_env=extra)
        start_worker("f", 50060, mock=True, model_id=args.model, latency_profile="a100_hf_fast",
                     profile_role="fast", vram_mb=20000)
        start_worker("s", 50061, mock=True, model_id=args.model, latency_profile="t4_emulated_slow",
                     profile_role="slow", vram_mb=12000)
        # VRAM pressure chaos on slow worker for -vram experiment mid-run optional
        if not wait_workers(2, 60):
            log(f"  skip {strategy}/{abl}: workers failed")
            continue
        if abl == "no_vram":
            # Still run; hard admission off means long prompts may "succeed" on low VRAM
            http_json(f"{MANAGER_HTTP}/debug/chaos/vram", method="POST",
                      body={"worker_id": "s", "free_vram_mb": 1500}, timeout=5.0)
        lr = run_load(f"n4_{strategy}_{abl}", duration_s=max(30.0, args.novelty_duration * 0.7),
                      concurrency=6, prompts=prompts)
        d = lr.to_dict()
        d["strategy"] = strategy
        d["nlms_mode"] = mode
        d["ablation"] = abl
        results.append(d)
        log(f"  {strategy}/{abl}: p99={d['p99_ms']} fail={d['n_fail']} mape={d['prediction_summary'].get('mape_pct')}")
    cleanup_all()
    return results


def exp_control_plane(args) -> List[dict]:
    """N5: Many mock workers — control plane stays healthy."""
    log("=== N5 Control-plane scalability ===")
    n = args.scale_workers
    cleanup_all()
    start_manager(strategy="NLMS", nlms_mode="DUAL", ablation="full",
                  slo_ms=90000, admission_off=True)
    for i in range(n):
        start_worker(
            f"m{i}", 50100 + i, mock=True, model_id=args.model,
            latency_profile="scalability_fast", vram_mb=4000, mock_seed=i,
        )
        if i % 8 == 7:
            time.sleep(0.2)
    if not wait_workers(n, timeout=120):
        log(f"WARNING: only partial workers registered for scale={n}")
    prompts = [{"prompt": f"scale {i}", "tier": "small"} for i in range(50)]
    lr = run_load("n5_scale", duration_s=30.0, concurrency=20, prompts=prompts)
    d = lr.to_dict()
    d["workers"] = n
    metrics = fetch_metrics()
    d["worker_count_reported"] = (metrics.get("workers") and len(metrics["workers"])) or 0
    cleanup_all()
    log(f"  scale n={n}: rps={d['rps']:.1f} p99={d['p99_ms']}")
    return [d]


def exp_main_e2e(args, prompts: List[dict]) -> List[dict]:
    """E1: Main strategy × (optional) real GPU comparison."""
    log("=== E1 Main e2e matrix ===")
    results = []
    strategies = ["NLMS", "RoundRobin", "LeastLoaded", "RLS", "STATIC"]
    if args.quick:
        strategies = ["NLMS", "RoundRobin", "LeastLoaded"]

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else []
    use_real = args.mode == "real" and len(gpus) >= 1

    for strategy in strategies:
        for seed in range(args.seeds):
            cleanup_all()
            extra = {"STATIC_SLOPE": "12.0", "STATIC_INTERCEPT": "200"} if strategy == "STATIC" else {}
            start_manager(strategy=strategy, nlms_mode="DUAL", ablation="full",
                          slo_ms=args.slo_ms, admission_off=True, extra_env=extra)
            if use_real:
                for i, dev in enumerate(gpus):
                    start_worker(
                        f"real{i}", 50060 + i, mock=False, model_id=args.model,
                        cuda_device=dev, tier="small", vram_mb=args.vram_mb,
                    )
                if not wait_workers(len(gpus), timeout=600):
                    log(f"  FAIL real workers strategy={strategy}")
                    continue
                # model load warm-up
                time.sleep(args.warmup_s)
            else:
                start_worker("fast", 50060, mock=True, model_id=args.model,
                             latency_profile="a100_hf_fast", vram_mb=20000)
                start_worker("slow", 50061, mock=True, model_id=args.model,
                             latency_profile="t4_emulated_slow", vram_mb=15000)
                if not wait_workers(2, 60):
                    continue
            lr = run_load(
                f"e1_{strategy}_s{seed}",
                duration_s=args.e2e_duration,
                concurrency=args.concurrency,
                prompts=prompts,
            )
            d = lr.to_dict()
            d["strategy"] = strategy
            d["seed"] = seed
            d["real_gpu"] = use_real
            d["gpus"] = gpus if use_real else ["mock", "mock"]
            results.append(d)
            log(f"  {strategy} seed={seed}: p99={d['p99_ms']} rps={d['rps']:.3f} ok={d['n_ok']}")
    cleanup_all()
    return results


def exp_vram_admission_realism(args, prompts: List[dict]) -> List[dict]:
    """N2b: Hard VRAM admission — chaos inject low VRAM, long prompts should 503."""
    log("=== N2b VRAM hard admission ===")
    cleanup_all()
    start_manager(strategy="NLMS", nlms_mode="DUAL", ablation="full",
                  slo_ms=90000, admission_off=False)
    start_worker("only", 50060, mock=True, model_id=args.model,
                 latency_profile="a100_hf_fast", vram_mb=3000)
    if not wait_workers(1, 60):
        return []
    http_json(f"{MANAGER_HTTP}/debug/chaos/vram", method="POST",
              body={"worker_id": "only", "free_vram_mb": 1500}, timeout=5.0)
    long_prompts = [{"prompt": "x" * 5000, "tier": "large"} for _ in range(40)]
    lr = run_load("n2b_vram", duration_s=20.0, concurrency=4, prompts=long_prompts, max_requests=40)
    d = lr.to_dict()
    cleanup_all()
    log(f"  vram: 503={d['n_503']} ok={d['n_ok']} reject_vram={d['admission'].get('rejected_vram')}")
    return [d]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def try_plot(summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed — skip figures")
        return

    FIGS.mkdir(parents=True, exist_ok=True)

    # N1 dual vs single MAPE
    n1 = summary.get("n1_dual_vs_single") or []
    if n1:
        labels = [r.get("nlms_mode", r["name"]) for r in n1]
        mapes = [r.get("prediction_summary", {}).get("mape_pct") or 0 for r in n1]
        p99s = [r.get("p99_ms") or 0 for r in n1]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        ax[0].bar(labels, mapes, color=["#2ca02c", "#d62728"])
        ax[0].set_ylabel("MAPE %")
        ax[0].set_title("N1 Prediction MAPE (lower better)")
        ax[1].bar(labels, p99s, color=["#2ca02c", "#d62728"])
        ax[1].set_ylabel("p99 latency (ms)")
        ax[1].set_title("N1 p99 under burst+thermal")
        fig.tight_layout()
        fig.savefig(FIGS / "n1_dual_vs_single.png", dpi=160)
        plt.close(fig)

    # N2 admission goodput
    n2 = summary.get("n2_admission") or []
    if n2:
        labels = ["off" if r.get("admission_off") else "on" for r in n2]
        gp = [r.get("goodput_under_slo_rps") or 0 for r in n2]
        p99 = [r.get("p99_ms") or 0 for r in n2]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        ax[0].bar(labels, gp, color=["#d62728", "#2ca02c"])
        ax[0].set_title("N2 Goodput under SLO (rps)")
        ax[1].bar(labels, p99, color=["#d62728", "#2ca02c"])
        ax[1].set_title("N2 p99 of completed")
        fig.tight_layout()
        fig.savefig(FIGS / "n2_admission_goodput.png", dpi=160)
        plt.close(fig)

    # N4 ablations p99
    n4 = summary.get("n4_ablations") or []
    if n4:
        labels = [f"{r.get('strategy','')}/{r.get('ablation','')}" for r in n4]
        p99 = [r.get("p99_ms") or 0 for r in n4]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4))
        ax.bar(range(len(labels)), p99, color="#1f77b4")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("p99 ms")
        ax.set_title("N4 Ablations / baselines")
        fig.tight_layout()
        fig.savefig(FIGS / "n4_ablations.png", dpi=160)
        plt.close(fig)

    # E1 strategies
    e1 = summary.get("e1_main") or []
    if e1:
        by_s: Dict[str, List[float]] = {}
        for r in e1:
            by_s.setdefault(r.get("strategy", "?"), []).append(r.get("p99_ms") or 0)
        labels = list(by_s.keys())
        means = [statistics.mean(v) for v in by_s.values()]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, means, color="#2ca02c")
        ax.set_ylabel("mean p99 ms")
        ax.set_title("E1 Main strategies")
        fig.tight_layout()
        fig.savefig(FIGS / "e1_main_p99.png", dpi=160)
        plt.close(fig)

    log(f"Figures written to {FIGS}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DIO camera-ready experiment suite")
    p.add_argument("--mode", choices=["mock", "real"], default="mock",
                   help="mock = calibrated latency workers; real = load HF model on GPUs")
    p.add_argument("--gpus", default="", help="Comma-separated CUDA devices for --mode real (e.g. 0,1)")
    p.add_argument("--model", default=os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct"))
    p.add_argument("--only", choices=["all", "novelty", "e2e", "scale"], default="all")
    p.add_argument("--quick", action="store_true", help="Shorter durations for smoke")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--slo-ms", type=float, default=5000.0,
                   help="Admission/goodput SLO in ms (raise for long real decode)")
    p.add_argument("--novelty-duration", type=float, default=45.0)
    p.add_argument("--e2e-duration", type=float, default=90.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--overload-rps", type=float, default=25.0)
    p.add_argument("--scale-workers", type=int, default=16)
    p.add_argument("--vram-mb", type=int, default=14000)
    p.add_argument("--warmup-s", type=float, default=40.0)
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.novelty_duration = min(args.novelty_duration, 25.0)
        args.e2e_duration = min(args.e2e_duration, 40.0)
        args.scale_workers = min(args.scale_workers, 8)
        args.seeds = 1

    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    log(f"DIO camera-ready suite mode={args.mode} only={args.only} root={DIO_ROOT}")
    os.environ["MODEL_ID"] = args.model

    if not args.skip_build:
        build_manager(force=False)

    prompts = make_prompts(300)
    summary: Dict[str, Any] = {
        "generated_at": utc_now(),
        "mode": args.mode,
        "model": args.model,
        "args": {k: getattr(args, k) for k in vars(args)},
        "novelty_claims": [
            "dual-timescale NLMS vs single-µ under burst+thermal",
            "admission reject if min S_w > SLO improves goodput under overload",
            "joint tier+VRAM+latency cost for multi-model",
            "ablations of cost terms + STATIC/RLS/RR/LL baselines",
        ],
    }

    try:
        if args.only in ("all", "novelty"):
            summary["n1_dual_vs_single"] = exp_dual_vs_single(args, prompts)
            summary["n2_admission"] = exp_admission_goodput(args, prompts)
            summary["n2b_vram"] = exp_vram_admission_realism(args, prompts)
            summary["n3_multi_tier"] = exp_multi_tier(args, prompts)
            summary["n4_ablations"] = exp_ablations(args, prompts)

        if args.only in ("all", "scale"):
            summary["n5_control_plane"] = exp_control_plane(args)

        if args.only in ("all", "e2e"):
            summary["e1_main"] = exp_main_e2e(args, prompts)

    except KeyboardInterrupt:
        log("Interrupted")
    except Exception as e:
        log(f"FATAL: {e}")
        traceback.print_exc()
        summary["error"] = str(e)
    finally:
        cleanup_all()

    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log(f"Wrote {out}")

    # Also drop a paper-friendly CSV-ish table
    rows = []
    for key in ("n1_dual_vs_single", "n2_admission", "n3_multi_tier", "n4_ablations", "e1_main"):
        for r in summary.get(key) or []:
            rows.append({"suite": key, **{k: r.get(k) for k in (
                "name", "strategy", "nlms_mode", "ablation", "p50_ms", "p99_ms",
                "rps", "goodput_under_slo_rps", "n_ok", "n_503", "n_fail")}})
    (RESULTS / "tables.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    try_plot(summary)

    # Console digest
    print("\n" + "=" * 60)
    print("CAMERA-READY DIGEST")
    print("=" * 60)
    for key in ("n1_dual_vs_single", "n2_admission", "n4_ablations", "e1_main"):
        block = summary.get(key) or []
        if not block:
            continue
        print(f"\n[{key}]")
        for r in block:
            print(f"  {r.get('name')}: p99={r.get('p99_ms')} rps={r.get('rps')} "
                  f"503={r.get('n_503')} mape={r.get('prediction_summary', {}).get('mape_pct')}")
    print(f"\nFull JSON: {out}")
    print(f"Figures:   {FIGS}")
    print("Done.")
    return 0 if "error" not in summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
