#!/usr/bin/env python3
"""
End-to-end REAL validation of DIO on this machine.

1. Starts 1–2 real HF transformers OpenAI-compatible engines (not mock sleep).
2. Starts DIO gateway in front.
3. Sends chat requests, checks routing headers + completions.
4. Compares NLMS vs RoundRobin briefly.
5. Writes results JSON for PERFORMANCE.md.

Usage:
  cd dio-serve
  pip install -e .
  python scripts/validate_local_realtime.py
  python scripts/validate_local_realtime.py --model Qwen/Qwen2.5-0.5B-Instruct --n 12
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_validation"
ENGINE = ROOT / "scripts" / "real_engine_server.py"
PY = sys.executable


def wait_url(url: str, timeout: float = 300.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def start_engine(port: int, model: str, mult: float, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    cmd = [
        PY,
        str(ENGINE),
        "--port",
        str(port),
        "--model",
        model,
        "--latency-mult",
        str(mult),
        "--host",
        "127.0.0.1",
    ]
    print(f"  starting engine :{port} mult={mult} model={model}", flush=True)
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))


def start_dio(backends: list[str], port: int, strategy: str, log_path: Path) -> subprocess.Popen:
    """Launch dio CLI serve."""
    f = open(log_path, "w", encoding="utf-8")
    cmd = [PY, "-m", "dio", "serve", "--port", str(port), "--strategy", strategy, "--admission-off"]
    for i, b in enumerate(backends):
        cmd.extend(["-b", f"e{i}={b}"])
    print(f"  starting DIO :{port} strategy={strategy} backends={backends}", flush=True)
    env = os.environ.copy()
    env["DIO_ADMISSION_OFF"] = "1"
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)


def kill(proc: subprocess.Popen | None) -> None:
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
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def chat(client: httpx.Client, base: str, content: str, max_tokens: int = 24) -> dict:
    t0 = time.perf_counter()
    r = client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
        },
        timeout=180.0,
    )
    ms = (time.perf_counter() - t0) * 1000
    return {
        "status": r.status_code,
        "ms": ms,
        "backend": r.headers.get("X-DIO-Backend"),
        "body": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:200],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=10, help="requests per strategy")
    ap.add_argument("--single-engine", action="store_true", help="only one real engine (smaller VRAM)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "python": PY,
    }

    try:
        # Device probe via a short import in child engines; also print host torch
        import torch

        results["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        print("Torch:", results["torch"], flush=True)

        # --- Engines ---
        e0 = start_engine(18000, args.model, 1.0, OUT / "engine0.log")
        procs.append(e0)
        if not wait_url("http://127.0.0.1:18000/health", timeout=600):
            print("FAIL: engine0 did not start. See results_validation/engine0.log")
            print((OUT / "engine0.log").read_text(encoding="utf-8", errors="replace")[-2000:])
            return 1
        h0 = httpx.get("http://127.0.0.1:18000/health", timeout=5).json()
        print("Engine0 health:", h0, flush=True)
        results["engine0_health"] = h0
        backends = ["http://127.0.0.1:18000"]

        if not args.single_engine:
            # Second engine: same model, artificial 2x sleep after real decode (still real tokens)
            e1 = start_engine(18001, args.model, 2.0, OUT / "engine1.log")
            procs.append(e1)
            if wait_url("http://127.0.0.1:18001/health", timeout=600):
                backends.append("http://127.0.0.1:18001")
                results["engine1_health"] = httpx.get("http://127.0.0.1:18001/health", timeout=5).json()
                print("Engine1 health:", results["engine1_health"], flush=True)
            else:
                print("WARN: second engine failed; continuing single-engine", flush=True)

        # Direct engine smoke (bypass DIO)
        with httpx.Client() as c:
            d = chat(c, "http://127.0.0.1:18000", "Say hello in 3 words.", max_tokens=16)
            print("Direct engine:", d["status"], d["ms"], str(d["body"])[:120], flush=True)
            results["direct_engine"] = {
                "status": d["status"],
                "ms": d["ms"],
                "ok": d["status"] == 200,
                "sample": str(d["body"])[:300],
            }
            if d["status"] != 200:
                print("FAIL: direct engine chat failed")
                return 1

        # --- DIO + NLMS ---
        dio_nlms = start_dio(backends, 18085, "nlms", OUT / "dio_nlms.log")
        procs.append(dio_nlms)
        if not wait_url("http://127.0.0.1:18085/healthz", timeout=60):
            print("FAIL: DIO NLMS did not start")
            print((OUT / "dio_nlms.log").read_text(encoding="utf-8", errors="replace")[-1500:])
            return 1

        routes: dict[str, int] = {}
        lats = []
        texts = []
        with httpx.Client() as c:
            for i in range(args.n):
                r = chat(c, "http://127.0.0.1:18085", f"Question {i}: what is 2+2? Answer briefly.", max_tokens=20)
                print(f"  NLMS #{i}: status={r['status']} backend={r['backend']} ms={r['ms']:.0f}", flush=True)
                if r["status"] != 200:
                    results["nlms_error"] = r
                    print("FAIL: DIO NLMS request failed", r)
                    return 1
                lats.append(r["ms"])
                b = r["backend"] or "?"
                routes[b] = routes.get(b, 0) + 1
                try:
                    texts.append(r["body"]["choices"][0]["message"]["content"][:80])
                except Exception:
                    texts.append(str(r["body"])[:80])

        metrics = httpx.get("http://127.0.0.1:18085/debug/metrics", timeout=10).json()
        results["nlms"] = {
            "n": args.n,
            "routes": routes,
            "p50_ms": sorted(lats)[len(lats) // 2],
            "p99_ms": sorted(lats)[max(0, int(0.99 * (len(lats) - 1)))],
            "mean_ms": sum(lats) / len(lats),
            "samples": texts[:3],
            "workers": metrics.get("workers"),
            "prediction": metrics.get("prediction"),
        }
        print("NLMS routes:", routes, "p50:", results["nlms"]["p50_ms"], flush=True)

        kill(dio_nlms)
        procs.remove(dio_nlms)
        time.sleep(1)

        # --- DIO + RoundRobin (if multi-engine) ---
        if len(backends) > 1:
            dio_rr = start_dio(backends, 18086, "round_robin", OUT / "dio_rr.log")
            procs.append(dio_rr)
            if wait_url("http://127.0.0.1:18086/healthz", timeout=60):
                rr_routes: dict[str, int] = {}
                rr_lats = []
                with httpx.Client() as c:
                    for i in range(args.n):
                        r = chat(c, "http://127.0.0.1:18086", f"RR {i}: say OK", max_tokens=8)
                        if r["status"] == 200:
                            rr_lats.append(r["ms"])
                            b = r["backend"] or "?"
                            rr_routes[b] = rr_routes.get(b, 0) + 1
                results["round_robin"] = {
                    "routes": rr_routes,
                    "p50_ms": sorted(rr_lats)[len(rr_lats) // 2] if rr_lats else None,
                    "mean_ms": sum(rr_lats) / len(rr_lats) if rr_lats else None,
                }
                print("RR routes:", rr_routes, flush=True)
            kill(dio_rr)
            if dio_rr in procs:
                procs.remove(dio_rr)

        results["pass"] = True
        results["summary"] = (
            f"REAL e2e OK on {h0.get('device')} | model={args.model} | "
            f"DIO NLMS routed {routes} | p50={results['nlms']['p50_ms']:.0f}ms"
        )
        print("\n=== VALIDATION PASS ===", flush=True)
        print(results["summary"], flush=True)

    except Exception as e:
        results["pass"] = False
        results["error"] = str(e)
        print("FAIL:", e, flush=True)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        for p in procs:
            kill(p)
        out_path = OUT / "local_realtime.json"
        out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print("Wrote", out_path, flush=True)

    return 0 if results.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
