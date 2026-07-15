#!/usr/bin/env python3
"""
Standalone multi-seed real dual-GPU heterogeneity test (T2-style).

Works even if run_gpu_cluster_validation.py is an older revision without
--hetero-seeds / delay-proxy. Starts (or reuses) two engines, wraps e1 with a
latency delay proxy, runs NLMS vs RR for N seeds, writes mean±std JSON.

Notebook (Kaggle):
  %run scripts/run_real_hetero_multiseed.py --engine-mode vllm --gpus 0,1 \\
      --model Qwen/Qwen2.5-3B-Instruct --seeds 5 --requests-per-seed 30

Or pure Python:
  import runpy; runpy.run_path("scripts/run_real_hetero_multiseed.py")
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
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    print("pip install httpx", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENGINE_HF = ROOT / "scripts" / "real_engine_server.py"
DELAY_PROXY = ROOT / "scripts" / "latency_delay_proxy.py"
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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mean_std(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"mean": m, "std": s, "n": len(xs)}


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


class Session:
    def __init__(self, out: Path):
        self.out = out
        self.logs = out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.handles: List[subprocess.Popen] = []
        self._files = []

    def start(self, name: str, cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
        log_path = self.logs / f"{name}.log"
        f = open(log_path, "w", encoding="utf-8")
        self._files.append(f)
        e = os.environ.copy()
        if env:
            e.update(env)
        kwargs: Dict[str, Any] = {"stdout": f, "stderr": subprocess.STDOUT, "cwd": str(ROOT), "env": e}
        if os.name != "nt":
            kwargs["preexec_fn"] = os.setsid
        log(f"START {name}: {' '.join(cmd[:14])}...")
        p = subprocess.Popen(cmd, **kwargs)
        self.handles.append(p)
        return p

    def cleanup(self) -> None:
        for p in reversed(self.handles):
            kill_proc(p)
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self.handles.clear()


def ensure_delay_proxy_script() -> None:
    """Write latency_delay_proxy.py if missing (older Kaggle checkouts)."""
    if DELAY_PROXY.exists():
        return
    DELAY_PROXY.write_text(
        '''#!/usr/bin/env python3
from __future__ import annotations
import argparse, time
from typing import Any, Dict
import httpx, uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

def build_app(upstream: str, latency_mult: float) -> FastAPI:
    upstream = upstream.rstrip("/")
    mult = max(1.0, float(latency_mult))
    app = FastAPI(title="dio-latency-delay-proxy")
    client = httpx.Client(timeout=300.0)
    def _proxy(path: str, request: Request, body: bytes) -> Response:
        t0 = time.perf_counter()
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length", "transfer-encoding")}
        try:
            r = client.request(request.method, f"{upstream}{path}", content=body,
                               headers=headers, params=dict(request.query_params))
        except Exception as e:
            return JSONResponse({"error": f"upstream: {e}"}, status_code=502)
        elapsed = time.perf_counter() - t0
        if mult > 1.0:
            time.sleep(elapsed * (mult - 1.0))
        out_headers = {k: v for k, v in r.headers.items()
                       if k.lower() not in ("content-encoding", "transfer-encoding",
                                            "content-length", "connection")}
        out_headers["X-DIO-Delay-Mult"] = str(mult)
        return Response(content=r.content, status_code=r.status_code,
                        headers=out_headers, media_type=r.headers.get("content-type"))
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def catch_all(path: str, request: Request):
        return _proxy("/" + path if path else "/", request, await request.body())
    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "upstream": upstream, "latency_mult": mult}
    return app

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18101)
    ap.add_argument("--latency-mult", type=float, default=2.0)
    args = ap.parse_args()
    uvicorn.run(build_app(args.upstream, args.latency_mult),
                host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )
    log(f"Wrote missing {DELAY_PROXY}")


def start_vllm(session: Session, gpu: str, port: int, model: str, max_model_len: int, gpu_mem_util: float, name: str):
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--host", "127.0.0.1", "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
    ]
    return session.start(name, cmd, env={"CUDA_VISIBLE_DEVICES": str(gpu)})


def start_dio(session: Session, backends: List[str], port: int, strategy: str, name: str):
    cmd = [
        PY, "-m", "dio", "serve",
        "--port", str(port), "--host", "127.0.0.1",
        "--strategy", strategy, "--nlms-mode", "dual",
        "--slo-ms", "180000", "--admission-off",
    ]
    for i, b in enumerate(backends):
        cmd.extend(["-b", f"e{i}={b}"])
    return session.start(name, cmd)


def run_load(base: str, model: str, n: int, max_tokens: int, seed: int) -> Dict[str, Any]:
    random.seed(seed)
    e2e: List[float] = []
    routes: Dict[str, int] = {}
    ok = fail = 0
    with httpx.Client(timeout=300.0) as client:
        try:
            h = client.get(f"{base.rstrip('/')}/healthz")
            health = h.json() if h.status_code == 200 else {"status": h.status_code}
        except Exception as e:
            return {"error": str(e), "ok": 0, "fail": n}
        for i in range(n):
            prompt = PROMPTS[i % len(PROMPTS)] + f" (seed={seed} i={i})"
            t0 = time.perf_counter()
            try:
                r = client.post(
                    f"{base.rstrip('/')}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                    },
                    headers={"X-DIO-Tier": "small"},
                )
                ms = (time.perf_counter() - t0) * 1000.0
                if r.status_code == 200:
                    ok += 1
                    e2e.append(ms)
                    wid = r.headers.get("X-DIO-Backend") or "unknown"
                    routes[wid] = routes.get(wid, 0) + 1
                else:
                    fail += 1
            except Exception:
                fail += 1
        # prediction metrics from gateway if available
        pred = {}
        try:
            m = client.get(f"{base.rstrip('/')}/debug/metrics").json()
            # aggregate worker MAPE if present
            mapes = []
            for w in (m.get("workers") or {}).values():
                if isinstance(w, dict) and w.get("mape_pct") is not None:
                    mapes.append(float(w["mape_pct"]))
            if mapes:
                pred = {"mape_pct": statistics.mean(mapes), "workers": m.get("workers")}
            else:
                pred = {"raw_keys": list(m.keys())[:12]}
        except Exception as e:
            pred = {"error": str(e)}

    def pct(xs: List[float], p: float) -> Optional[float]:
        if not xs:
            return None
        ys = sorted(xs)
        k = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
        return ys[k]

    total = sum(routes.values()) or 1
    return {
        "health": health,
        "n": n,
        "ok": ok,
        "fail": fail,
        "routes": routes,
        "frac_e0": routes.get("e0", 0) / total,
        "e2e_p50_ms": pct(e2e, 50),
        "e2e_p99_ms": pct(e2e, 99),
        "e2e_mean_ms": statistics.mean(e2e) if e2e else None,
        "prediction": pred,
        "seed": seed,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Real dual-GPU hetero multi-seed (delay proxy)")
    p.add_argument("--out", default=str(ROOT / "results_gpu_cluster_hetero"))
    p.add_argument("--engine-mode", choices=["vllm", "external"], default="vllm")
    p.add_argument("--backends", default="", help="external: url0,url1")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--requests-per-seed", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--engine-base-port", type=int, default=18000)
    p.add_argument("--proxy-port", type=int, default=18101)
    p.add_argument("--dio-base-port", type=int, default=19200)
    p.add_argument("--slow-mult", type=float, default=2.0)
    # allow notebook `%run` without blowing up on unknown trailing junk
    args, _unknown = p.parse_known_args()
    return args


def main() -> int:
    args = parse_args()
    ensure_delay_proxy_script()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = Session(out)
    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "run_real_hetero_multiseed.py",
        "args": vars(args),
        "purpose": "Real dual-GPU slope-skew via delay proxy; multi-seed NLMS vs RR",
    }
    backends: List[str] = []
    try:
        if args.engine_mode == "external" or args.backends.strip():
            backends = [b.strip() for b in args.backends.split(",") if b.strip()]
            if len(backends) < 2:
                log("ERROR: need two --backends for external mode")
                return 2
            for b in backends:
                wait_url(b.rstrip("/") + "/v1/models", timeout=60)
        else:
            gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
            if len(gpus) < 2:
                log("ERROR: need --gpus 0,1 for dual real-GPU hetero")
                return 2
            for i, gpu in enumerate(gpus[:2]):
                port = args.engine_base_port + i
                start_vllm(
                    session, gpu, port, args.model,
                    args.max_model_len, args.gpu_mem_util, f"vllm_gpu{gpu}",
                )
                backends.append(f"http://127.0.0.1:{port}")
            for b in backends:
                log(f"Waiting for {b} (vLLM may take several minutes)...")
                if not wait_url(b + "/v1/models", timeout=900):
                    log(f"ERROR: engine failed: {b}")
                    return 1

        # wrap e1 with delay proxy
        raw_e1 = backends[1]
        proxy_url = f"http://127.0.0.1:{args.proxy_port}"
        session.start(
            "delay_proxy_slow",
            [
                PY, str(DELAY_PROXY),
                "--upstream", raw_e1,
                "--host", "127.0.0.1",
                "--port", str(args.proxy_port),
                "--latency-mult", str(args.slow_mult),
            ],
        )
        if not wait_url(proxy_url + "/health", timeout=60):
            log("ERROR: delay proxy failed to start")
            return 1
        hetero_backends = [backends[0], proxy_url]
        summary["backends_raw"] = backends
        summary["backends_hetero"] = hetero_backends
        summary["note"] = (
            f"e0=fast raw engine; e1=real engine {raw_e1} behind delay proxy ×{args.slow_mult}"
        )
        log(summary["note"])

        rows: Dict[str, List[dict]] = {"nlms": [], "round_robin": []}
        for strat in ("nlms", "round_robin"):
            for seed in range(args.seeds):
                port = args.dio_base_port + (0 if strat == "nlms" else 50) + seed
                start_dio(session, hetero_backends, port, strat, f"dio_{strat}_s{seed}")
                url = f"http://127.0.0.1:{port}"
                if not wait_url(url + "/healthz", timeout=90):
                    rows[strat].append({"seed": seed, "error": "dio_start_failed"})
                    continue
                row = run_load(url, args.model, args.requests_per_seed, args.max_tokens, seed=1000 + seed)
                row["seed"] = seed
                row["strategy"] = strat
                rows[strat].append(row)
                log(
                    f"  {strat} seed={seed}: frac_e0={row.get('frac_e0')} "
                    f"p99={row.get('e2e_p99_ms')} ok={row.get('ok')}/{row.get('n')}"
                )
                # kill last dio only (engines stay up)
                if session.handles:
                    kill_proc(session.handles.pop())
                time.sleep(0.5)

        def agg_field(strat: str, field: str):
            xs = [float(r[field]) for r in rows[strat] if r.get(field) is not None]
            return mean_std(xs)

        imps = []
        for i in range(min(len(rows["nlms"]), len(rows["round_robin"]))):
            a, b = rows["nlms"][i], rows["round_robin"][i]
            if a.get("e2e_p99_ms") and b.get("e2e_p99_ms") and b["e2e_p99_ms"] > 0:
                imps.append((b["e2e_p99_ms"] - a["e2e_p99_ms"]) / b["e2e_p99_ms"] * 100.0)

        summary["G3_hetero"] = {
            "seeds": args.seeds,
            "n_per_seed": args.requests_per_seed,
            "slow_mult": args.slow_mult,
            "nlms_frac_e0": agg_field("nlms", "frac_e0"),
            "rr_frac_e0": agg_field("round_robin", "frac_e0"),
            "nlms_p99": agg_field("nlms", "e2e_p99_ms"),
            "rr_p99": agg_field("round_robin", "e2e_p99_ms"),
            "p99_improvement_pct": mean_std(imps),
            "per_seed": rows,
        }
        summary["status"] = "ok"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    except Exception as e:
        summary["status"] = "error"
        summary["error"] = str(e)
        import traceback
        traceback.print_exc()
    finally:
        session.cleanup()

    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    g3 = summary.get("G3_hetero") or {}
    snip = [
        "# Real dual-GPU hetero (delay proxy) multi-seed",
        f"Generated: {summary.get('generated_at')}",
        f"slow_mult={args.slow_mult} seeds={args.seeds}",
        "",
        f"- NLMS frac to fast (e0): {g3.get('nlms_frac_e0')}",
        f"- RR   frac to e0:       {g3.get('rr_frac_e0')}",
        f"- NLMS p99:              {g3.get('nlms_p99')}",
        f"- RR   p99:              {g3.get('rr_p99')}",
        f"- p99 improvement %:     {g3.get('p99_improvement_pct')}",
        "",
        f"Full JSON: {path}",
    ]
    (out / "paper_snippets.md").write_text("\n".join(snip) + "\n", encoding="utf-8")
    print("\n" + "=" * 60)
    print("REAL HETERO MULTI-SEED COMPLETE")
    print(f"Status: {summary.get('status')}")
    if g3:
        print(f"  NLMS frac_e0: {g3.get('nlms_frac_e0')}")
        print(f"  p99 impr %:   {g3.get('p99_improvement_pct')}")
    print(f"Results → {out}")
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
