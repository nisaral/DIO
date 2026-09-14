#!/usr/bin/env python3
"""
Regime D — physically heterogeneous GPU workers (no injected delay).

Unlike Regime C (identical T4s + latency_delay_proxy x2), Regime D points DIO at
real, different GPU SKUs (e.g. L4 24GB + A100 40GB), each running stock vLLM.
DIO sees only OpenAI-compatible HTTP + Prometheus /metrics — zero engine changes.

Three cells:
  D1  Strategy head-to-head: nlms vs rls vs round_robin vs least_loaded
      Mixed output lengths so the token feature N actually varies, which is what
      makes the slope in y ~= s*N + b identifiable.
  D2  Slope identifiability: per-SKU learned NLMS slope vs offline OLS fit on the
      same observed (N, latency) pairs. This is the "DIO discovers the A100 is
      faster" figure, validated rather than asserted.
  D3  max_tokens sensitivity: repeat D1 at several fixed max_tokens to show that
      short-decode regimes (32) under-identify the slope vs long decode (256+).

Usage (gateway node, engines already up on the workers):
  python scripts/run_regime_d_hetero.py \\
      --backends l4=http://10.0.0.11:8000,a100=http://10.0.0.12:8000 \\
      --vram l4=24000,a100=40000 \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --tokenizer Qwen/Qwen2.5-3B-Instruct \\
      --seeds 10 --requests-per-seed 40

Cheap dry-run with no GPUs (two mock engines, different speeds):
  python scripts/run_regime_d_hetero.py --mock --seeds 3 --requests-per-seed 12
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
except ImportError:  # pragma: no cover
    print("pip install httpx", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MOCK_SERVER = ROOT / "scripts" / "mock_vllm_server.py"

STRATEGIES = ("nlms", "rls", "round_robin", "least_loaded")

# Prompt bodies of differing length; combined with varied max_tokens this gives
# the spread in N that the slope estimate needs.
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


def mean_std(xs: List[float]) -> Dict[str, Any]:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (max(0.0, min(100.0, p)) / 100.0) * (len(ys) - 1)
    lo, hi = int(k), min(len(ys) - 1, int(k) + 1)
    if lo == hi:
        return ys[lo]
    t = k - lo
    return ys[lo] * (1.0 - t) + ys[hi] * t


def ols_fit(pairs: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Offline least squares y = s*x + b. Ground truth for the learned slope."""
    pts = [(float(x), float(y)) for x, y in pairs if x and x > 0]
    n = len(pts)
    if n < 3:
        return {"slope": None, "intercept": None, "n": n, "r2": None}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        # No spread in N -> slope is not identifiable. Say so instead of guessing.
        return {"slope": None, "intercept": None, "n": n, "r2": None,
                "note": "degenerate: no variance in token feature"}
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in pts)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None
    return {"slope": slope, "intercept": intercept, "n": n, "r2": r2,
            "token_spread": {"min": min(xs), "max": max(xs), "std": statistics.stdev(xs) if n > 1 else 0.0}}


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


def wait_url(url: str, timeout: float = 180.0, interval: float = 1.5) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=3.0).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


class Session:
    def __init__(self, out: Path) -> None:
        self.logs = out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.handles: List[subprocess.Popen] = []
        self._files: List[Any] = []

    def start(self, name: str, cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
        f = open(self.logs / f"{name}.log", "w", encoding="utf-8")
        self._files.append(f)
        e = os.environ.copy()
        # The gateway banner contains non-ASCII (an arrow). With stdout piped to
        # a file, Windows defaults the child's encoding to cp1252 and the banner
        # raises UnicodeEncodeError before the server ever binds -- so the whole
        # run dies at wait_url() with nothing but a timeout to show for it.
        # Harmless on Linux, where the default is already UTF-8.
        e.setdefault("PYTHONIOENCODING", "utf-8")
        if env:
            e.update(env)
        kwargs: Dict[str, Any] = {
            "stdout": f, "stderr": subprocess.STDOUT, "cwd": str(ROOT), "env": e,
        }
        if os.name != "nt":
            kwargs["preexec_fn"] = os.setsid
        log(f"START {name}")
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


def parse_kv(spec: str) -> List[Tuple[str, str]]:
    """'l4=http://a:8000,a100=http://b:8000' -> [('l4','http://a:8000'), ...]

    Bare URLs are accepted too and get positional ids e0, e1, ...
    """
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


def start_dio(
    session: Session,
    *,
    backends: List[Tuple[str, str]],
    vram: Dict[str, float],
    port: int,
    strategy: str,
    name: str,
    tokenizer: str = "",
    engine_metrics: bool = True,
) -> Tuple[str, subprocess.Popen]:
    """Launch the gateway with per-backend VRAM declared.

    Regime C's harness omitted --vram, so every worker inherited the 24000 MB
    default. On mixed SKUs that mis-declares the A100 and biases vram_cost in
    the joint score, so we pass it explicitly here.
    """
    cmd = [
        PY, "-m", "dio", "serve",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--strategy", strategy,
        "--nlms-mode", "dual",
        "--slo-ms", "180000",
        "--admission-off",
        "--admission-mode", "rank_only",
    ]
    cmd.append("--engine-metrics" if engine_metrics else "--no-engine-metrics")
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    # --vram is positional w.r.t. --backend, so append in the same order.
    for bid, url in backends:
        cmd.extend(["-b", f"{bid}={url}"])
    for bid, _ in backends:
        cmd.extend(["--vram", str(float(vram.get(bid, 24000.0)))])
    proc = session.start(name, cmd, env={"DIO_METRICS_INTERVAL_S": "0.5"})
    return f"http://127.0.0.1:{port}", proc


def build_workload(n: int, seed: int, max_tokens_choices: List[int]) -> List[Dict[str, Any]]:
    """Mixed prompt length x mixed max_tokens -> real spread in the token feature."""
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


def run_load(base: str, model: str, workload: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sequential load; records per-request routing + latency for OLS."""
    base = base.rstrip("/")
    e2e: List[float] = []
    routes: Dict[str, int] = {}
    per_req: List[Dict[str, Any]] = []
    ok = fail = 0

    with httpx.Client(timeout=600.0) as client:
        for item in workload:
            t0 = time.perf_counter()
            try:
                r = client.post(
                    f"{base}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": item["prompt"]}],
                        "max_tokens": item["max_tokens"],
                        "temperature": 0.0,
                    },
                    headers={"X-DIO-Tier": "small"},
                )
                ms = (time.perf_counter() - t0) * 1000.0
                if r.status_code != 200:
                    fail += 1
                    continue
                ok += 1
                e2e.append(ms)
                wid = r.headers.get("X-DIO-Backend") or "unknown"
                routes[wid] = routes.get(wid, 0) + 1
                # Token count as the gateway itself computed it, so the OLS fit
                # uses the same feature NLMS trained on.
                tokens = None
                try:
                    j = r.json()
                    dec = (j.get("dio") or {}).get("decision") or {}
                    tokens = dec.get("tokens")
                    usage = j.get("usage") or {}
                except Exception:
                    usage = {}
                per_req.append({
                    "worker": wid,
                    "e2e_ms": ms,
                    "tokens_feature": tokens,
                    "max_tokens": item["max_tokens"],
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                })
            except Exception:
                fail += 1

        workers_snap: Dict[str, Any] = {}
        pred_samples: List[Dict[str, Any]] = []
        try:
            m = client.get(f"{base}/debug/metrics", timeout=20.0).json()
            for wid, w in (m.get("workers") or {}).items():
                if isinstance(w, dict):
                    workers_snap[wid] = {
                        "fast_slope": w.get("fast_slope"),
                        "slow_slope": w.get("slow_slope"),
                        "intercept": w.get("intercept"),
                        "updates": w.get("updates"),
                        "mape_pct": w.get("mape_pct"),
                        "mae_ms": w.get("mae_ms"),
                        "avg_latency_ms": w.get("avg_latency_ms"),
                        "tier": w.get("tier"),
                        "free_vram_mb": w.get("free_vram_mb"),
                    }
            pred_samples = (m.get("prediction") or {}).get("samples") or []
        except Exception as e:
            workers_snap = {"error": str(e)}

    total = sum(routes.values()) or 1
    # Offline OLS per worker on (token feature, observed latency).
    ols: Dict[str, Any] = {}
    for wid in routes:
        pairs = [
            (r["tokens_feature"], r["e2e_ms"])
            for r in per_req
            if r["worker"] == wid and r.get("tokens_feature")
        ]
        ols[wid] = ols_fit(pairs)

    return {
        "n": len(workload),
        "ok": ok,
        "fail": fail,
        "routes": routes,
        "route_frac": {k: v / total for k, v in routes.items()},
        "e2e_p50_ms": percentile(e2e, 50),
        "e2e_p95_ms": percentile(e2e, 95),
        "e2e_p99_ms": percentile(e2e, 99),
        "e2e_mean_ms": statistics.mean(e2e) if e2e else None,
        "learned": workers_snap,
        "ols": ols,
        "pred_sample_count": len(pred_samples),
        "per_request": per_req,
    }


MOCK_SPECS = [
    # (id, port, latency_ms, declared_vram_mb)
    # mock_vllm_server decodes at latency_ms/100 per token and prefills at
    # latency_ms/2, so the A100 fixture is the fast one -- matching the real
    # SKU ordering (HBM2e ~1555 GB/s vs L4 GDDR6 ~300 GB/s).
    ("a100", 18102, 250.0, 40000.0),
    ("l4", 18101, 750.0, 24000.0),
]

# A third tier, used only by --mock3, to exercise the N>2 reporting path before
# paying for a third rented GPU. Latency is set between the two above so the
# fixture mirrors a mid-bandwidth card (L40S/A40 class) rather than a clone.
MOCK_SPECS_3 = MOCK_SPECS + [("l40s", 18103, 430.0, 48000.0)]


def start_mocks(session: Session, specs=None) -> Tuple[List[Tuple[str, str]], Dict[str, float]]:
    """Mock vLLM servers with different decode speeds (CI / no-GPU smoke)."""
    backends: List[Tuple[str, str]] = []
    vram: Dict[str, float] = {}
    for bid, port, latency_ms, vram_mb in (specs or MOCK_SPECS):
        session.start(
            f"mock_{bid}",
            [PY, str(MOCK_SERVER), "--port", str(port),
             "--latency-ms", str(latency_ms), "--model", "mock-model"],
        )
        backends.append((bid, f"http://127.0.0.1:{port}"))
        vram[bid] = vram_mb
    return backends, vram


def write_paper_snippets(out: Path, summary: Dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Regime D — real heterogeneous SKUs (multi-seed)",
        f"Generated: {summary.get('generated_at')}",
        "backends: " + ", ".join(
            f"{b['id']}={b['url']} ({b['vram_mb']:.0f}MB)"
            for b in (summary.get("backends") or [])
        ) + ("  [MOCK FIXTURE — not paper data]" if summary.get("mock") else ""),
        f"seeds={args.seeds} reqs/seed={args.requests_per_seed}",
        "",
    ]
    def fmt(ms: Optional[Dict[str, Any]], nd: int = 0) -> str:
        if not ms or ms.get("mean") is None:
            return "--"
        return f"${ms['mean']:.{nd}f} \\pm {(ms.get('std') or 0.0):.{nd}f}$"

    d1 = summary.get("D1_strategy_head_to_head") or {}
    d2 = summary.get("D2_slope_identifiability") or {}
    d3 = summary.get("D3_max_tokens_sensitivity") or {}

    if d1:
        lines.append("## D1 strategy head-to-head")
        for strat in ("nlms", "rls", "round_robin", "least_loaded"):
            row = d1.get(strat)
            if not row:
                continue
            frac = {k: round(v, 3) for k, v in (row.get("route_frac") or {}).items()}
            lines.append(f"- {strat}: p99={fmt(row.get('p99'))} p50={fmt(row.get('p50'))} frac={frac}")
        for key, label in (("p99_improvement_vs_rr", "nlms"),
                           ("p99_improvement_vs_rr_rls", "rls"),
                           ("p99_improvement_vs_rr_least_loaded", "least_loaded")):
            imp = d1.get(key)
            if imp and imp.get("mean") is not None:
                lines.append(f"- p99 improvement vs RR ({label}): "
                             f"{imp['mean']:.1f}% +/- {(imp.get('std') or 0):.1f}% "
                             f"(wins {imp.get('wins')}/{imp.get('of')})")
        lines.append("")
        lines.append("### LaTeX rows (tab:realD)")
        for strat in ("nlms", "rls", "round_robin", "least_loaded"):
            row = d1.get(strat)
            if not row:
                continue
            lines.append(f"{strat} & {fmt(row.get('p50'))} & {fmt(row.get('p95'))} "
                         f"& {fmt(row.get('p99'))} & {row.get('seeds_ok')} \\\\")
        lines.append("")

    if d2:
        lines.append("## D2 learned NLMS slope vs offline OLS (per SKU)")
        for wid, v in (d2.get("slopes_by_worker") or {}).items():
            o = v.get("ols") or {}
            learned = (v.get("learned") or {}).get("mean")
            note = f" [{o['note']}]" if o.get("note") else ""
            lines.append(
                f"- {wid} (vram={v.get('declared_vram_mb')}MB, n={v.get('samples')}): "
                f"learned={learned if learned is None else round(learned, 3)} "
                f"OLS={o.get('slope') if o.get('slope') is None else round(o['slope'], 3)} "
                f"r2={o.get('r2') if o.get('r2') is None else round(o['r2'], 3)}{note}"
            )
        ratio = d2.get("slope_ratio") or {}
        if ratio.get("pair"):
            lr, orr = ratio.get("learned_ratio"), ratio.get("ols_ratio")
            lines.append(f"- slope ratio {ratio['pair']}: "
                         f"learned={lr if lr is None else round(lr, 3)} "
                         f"OLS={orr if orr is None else round(orr, 3)}")
        lines.append("")

    if d3:
        lines.append("## D3 max_tokens sensitivity (slope identifiability)")
        for L, c in (d3.get("cells") or {}).items():
            sl = {k: (round(v["mean"], 3) if v and v.get("mean") is not None else None)
                  for k, v in (c.get("learned_slope") or {}).items()}
            lines.append(f"- max_tokens={L}: p99={fmt(c.get('p99'))} slopes={sl} "
                         f"degenerate_ols_seeds={c.get('ols_degenerate_seeds')}")
        lines.append("")

    lines.append(f"Full JSON: {out / 'summary.json'}")
    (out / "paper_snippets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime D: real heterogeneous SKU capacity study")
    p.add_argument("--backends", default="",
                   help="comma list: id=http://host:port  (e.g. l4=...,a100=...)")
    p.add_argument("--vram", default="",
                   help="comma list: id=MB (e.g. l4=24000,a100=40000); defaults to 24000")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--requests-per-seed", type=int, default=40)
    p.add_argument("--max-tokens-list", default="32,64,128,256",
                   help="comma list of max_tokens the workload mixes")
    p.add_argument("--strategies", default="nlms,rls,round_robin,least_loaded",
                   help="comma list; subset of nlms,rls,round_robin,least_loaded")
    p.add_argument("--d1", action="store_true", help="run strategy head-to-head cell (default on)")
    p.add_argument("--no-d1", action="store_true")
    p.add_argument("--d2", action="store_true", help="run slope identifiability cell")
    p.add_argument("--d3", action="store_true",
                   help="run max_tokens sensitivity cell (nlms only, fixed-length workloads)")
    p.add_argument("--d3-lengths", default="32,128,256",
                   help="max_tokens values for D3 (comma list)")
    p.add_argument("--d3-seeds", type=int, default=5)
    p.add_argument("--mock", action="store_true", help="spin two local mock engines instead of --backends")
    p.add_argument("--mock3", action="store_true",
                   help="spin THREE local mock engines (exercises the >2 SKU reporting path)")
    p.add_argument("--dio-base-port", type=int, default=19300)
    p.add_argument("--out", default=str(ROOT / "results_regime_d"))
    p.add_argument("--provider", default="e2e")
    p.add_argument("--region", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = Session(out)
    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "run_regime_d_hetero.py",
        "args": vars(args),
        "provider": args.provider,
        "region": args.region,
        "protocol": "n=10 seeds x 30-40 reqs, mixed max_tokens, fresh gateway per seed, "
                    "per-worker VRAM declared, learned slope vs offline OLS",
    }
    backends: List[Tuple[str, str]] = []
    vram: Dict[str, float] = {}
    try:
        if args.mock or args.mock3:
            backends, vram = start_mocks(
                session, MOCK_SPECS_3 if args.mock3 else MOCK_SPECS
            )
            # Heuristic token feature only: no HF download on a CI/laptop run.
            args.tokenizer = ""
            args.model = "mock-model"
            for bid, u in backends:
                if not wait_url(u + "/v1/models", timeout=60):
                    log(f"ERROR: mock {bid} failed to start"); return 1
            summary["mock"] = True
            log("mock engines up")
        else:
            backends = parse_kv(args.backends)
            if len(backends) < 2:
                log("ERROR: need >=2 backends (id=http://host:port) or --mock"); return 2
            if args.vram.strip():
                for k, v in parse_kv(args.vram):
                    vram[k] = float(v)
            for bid, url in backends:
                if not wait_url(url.rstrip("/") + "/v1/models", timeout=900):
                    log(f"ERROR: engine not reachable: {bid} {url}"); return 1
                log(f"backend {bid} -> {url} OK")
        summary["backends"] = [{"id": b, "url": u, "vram_mb": vram.get(b, 24000.0)} for b, u in backends]
        summary["worker_skus"] = {b: {"declared_vram_mb": vram.get(b, 24000.0)} for b, _ in backends}
        max_tokens = [int(x) for x in args.max_tokens_list.split(",") if x.strip()]
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        strategies = [s for s in strategies if s in STRATEGIES]

        if not args.no_d1:
            log("== D1: strategy head-to-head ==")
            summary["D1_strategy_head_to_head"] = run_d1(
                session, backends, vram, strategies, args, max_tokens
            )
        if args.d2:
            log("== D2: slope identifiability ==")
            summary["D2_slope_identifiability"] = run_d2(
                session, backends, vram, args, max_tokens
            )
        if args.d3:
            log("== D3: max_tokens sensitivity ==")
            summary["D3_max_tokens_sensitivity"] = run_d3(
                session, backends, vram, args
            )
        summary["status"] = "ok"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    except Exception as e:
        summary["status"] = "error"
        summary["error"] = str(e)
        import traceback; traceback.print_exc()
    finally:
        session.cleanup()

    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_paper_snippets(out, summary, args)
    print("\n" + "=" * 60)
    print(f"REGIME D COMPLETE — status: {summary.get('status')}")
    print(f"Results -> {out}")
    return 0 if summary.get("status") == "ok" else 1


def _one_seed(
    session: Session,
    backends: List[Tuple[str, str]],
    vram: Dict[str, float],
    *,
    strategy: str,
    seed: int,
    port: int,
    args: argparse.Namespace,
    workload: List[Dict[str, Any]],
    tag: str,
) -> Dict[str, Any]:
    """Fresh gateway -> load -> teardown. One (strategy, seed) cell."""
    url, proc = start_dio(
        session,
        backends=backends,
        vram=vram,
        port=port,
        strategy=strategy,
        name=f"dio_{tag}_{strategy}_s{seed}",
        tokenizer=args.tokenizer,
    )
    try:
        if not wait_url(url + "/healthz", timeout=120):
            return {"seed": seed, "strategy": strategy, "error": "gateway_start_failed"}
        row = run_load(url, args.model, workload)
        row["seed"] = seed
        row["strategy"] = strategy
        return row
    finally:
        # Kill only this gateway; the engines are long-lived and stay up.
        kill_proc(proc)
        if proc in session.handles:
            session.handles.remove(proc)
        time.sleep(0.4)


def run_d1(
    session: Session,
    backends: List[Tuple[str, str]],
    vram: Dict[str, float],
    strategies: List[str],
    args: argparse.Namespace,
    max_tokens: List[int],
) -> Dict[str, Any]:
    """Strategy head-to-head on mixed-length traffic, n seeds each."""
    rows: Dict[str, List[Dict[str, Any]]] = {s: [] for s in strategies}
    for si, strat in enumerate(strategies):
        for seed in range(args.seeds):
            # Same workload per seed across strategies -> paired comparison.
            workload = build_workload(args.requests_per_seed, 1000 + seed, max_tokens)
            port = args.dio_base_port + si * 50 + seed
            row = _one_seed(
                session, backends, vram,
                strategy=strat, seed=seed, port=port, args=args,
                workload=workload, tag="d1",
            )
            rows[strat].append(row)
            log(f"  d1 {strat} seed={seed}: p99={row.get('e2e_p99_ms')} "
                f"frac={row.get('route_frac')} ok={row.get('ok')}/{row.get('n')}")

    result: Dict[str, Any] = {"per_seed": rows}
    for strat in strategies:
        good = [r for r in rows[strat] if not r.get("error")]
        frac_acc: Dict[str, List[float]] = {}
        for r in good:
            for wid, f in (r.get("route_frac") or {}).items():
                frac_acc.setdefault(wid, []).append(f)
        result[strat] = {
            "p50": mean_std([r.get("e2e_p50_ms") for r in good]),
            "p95": mean_std([r.get("e2e_p95_ms") for r in good]),
            "p99": mean_std([r.get("e2e_p99_ms") for r in good]),
            "mean": mean_std([r.get("e2e_mean_ms") for r in good]),
            "route_frac": {k: statistics.mean(v) for k, v in frac_acc.items()},
            "route_frac_std": {
                k: (statistics.stdev(v) if len(v) > 1 else 0.0) for k, v in frac_acc.items()
            },
            "seeds_ok": len(good),
        }

    # Paired per-seed improvement vs RR (same workload per seed), plus a sign
    # count so a mean is not carried by one lucky seed.
    if "round_robin" in rows:
        for strat in strategies:
            if strat == "round_robin":
                continue
            imps, wins = [], 0
            for a, b in zip(rows[strat], rows["round_robin"]):
                pa, pb = a.get("e2e_p99_ms"), b.get("e2e_p99_ms")
                if pa and pb and pb > 0:
                    imp = (pb - pa) / pb * 100.0
                    imps.append(imp)
                    wins += 1 if imp > 0 else 0
            key = "p99_improvement_vs_rr" if strat == "nlms" else f"p99_improvement_vs_rr_{strat}"
            result[key] = {**mean_std(imps), "wins": wins, "of": len(imps)}
    return result


def run_d2(
    session: Session,
    backends: List[Tuple[str, str]],
    vram: Dict[str, float],
    args: argparse.Namespace,
    max_tokens: List[int],
) -> Dict[str, Any]:
    """
    Slope identifiability: is the learned per-worker slope real?

    Uses round_robin so both SKUs get comparable traffic at comparable token
    mixes (NLMS starves the slow worker, which would leave too few samples to
    fit). NLMS still learns per-worker models under RR, so learned vs OLS is a
    fair comparison of the same estimator on balanced data.
    """
    rows: List[Dict[str, Any]] = []
    for seed in range(args.seeds):
        workload = build_workload(args.requests_per_seed, 5000 + seed, max_tokens)
        row = _one_seed(
            session, backends, vram,
            strategy="round_robin", seed=seed,
            port=args.dio_base_port + 400 + seed, args=args,
            workload=workload, tag="d2",
        )
        rows.append(row)
        log(f"  d2 seed={seed}: ols={ {k: round(v['slope'], 3) if v.get('slope') else None for k, v in (row.get('ols') or {}).items()} }")

    by_worker: Dict[str, Any] = {}
    pooled: Dict[str, List[Tuple[float, float]]] = {}
    for r in rows:
        if r.get("error"):
            continue
        for req in r.get("per_request") or []:
            if req.get("tokens_feature"):
                pooled.setdefault(req["worker"], []).append(
                    (float(req["tokens_feature"]), float(req["e2e_ms"]))
                )
    for wid, pairs in pooled.items():
        learned_vals = [
            (r.get("learned") or {}).get(wid, {}).get("fast_slope")
            for r in rows if not r.get("error")
        ]
        learned_vals = [v for v in learned_vals if isinstance(v, (int, float))]
        by_worker[wid] = {
            "declared_vram_mb": vram.get(wid, 24000.0),
            "learned": mean_std(learned_vals),
            "ols": ols_fit(pairs),
            "samples": len(pairs),
        }

    # Ratio of slopes is the SKU capacity claim; report both estimates of it.
    wids = sorted(by_worker)
    ratio: Dict[str, Any] = {}
    if len(wids) == 2:
        a, b = wids
        la = (by_worker[a]["learned"] or {}).get("mean")
        lb = (by_worker[b]["learned"] or {}).get("mean")
        oa = (by_worker[a]["ols"] or {}).get("slope")
        ob = (by_worker[b]["ols"] or {}).get("slope")
        ratio = {
            "pair": f"{a}/{b}",
            "learned_ratio": (la / lb) if (la and lb) else None,
            "ols_ratio": (oa / ob) if (oa and ob) else None,
        }

    # With three or more SKUs a single pair no longer describes the fleet, so
    # normalise every slope against the slowest worker. Each entry then reads as
    # "cost per token relative to the slowest card", which is scale-free and is
    # what ranking consumes. Computed for any N>=2 so the 2-worker case can be
    # cross-checked against `ratio` above.
    norm: Dict[str, Any] = {}
    def _slope(w: str, kind: str):
        if kind == "learned":
            return (by_worker[w]["learned"] or {}).get("mean")
        return (by_worker[w]["ols"] or {}).get("slope")

    for kind in ("learned", "ols"):
        vals = {w: _slope(w, kind) for w in wids}
        usable = {w: v for w, v in vals.items() if isinstance(v, (int, float)) and v > 0}
        if len(usable) < 2:
            continue
        ref = max(usable, key=lambda w: usable[w])  # slowest = largest ms/token
        norm[kind] = {
            "reference": ref,
            "ratios": {w: usable[w] / usable[ref] for w in usable},
            # Ordering is the claim that actually matters; record it explicitly
            # so a reader can compare the two estimators without recomputing.
            "order_fast_to_slow": sorted(usable, key=lambda w: usable[w]),
        }
    if norm:
        ratio["normalized"] = norm
    # per_request is retained for OLS above; drop it from the dump to keep
    # summary.json small enough to read by hand.
    for r in rows:
        r.pop("per_request", None)
    return {"slopes_by_worker": by_worker, "slope_ratio": ratio, "per_seed": rows}


def run_d3(
    session: Session,
    backends: List[Tuple[str, str]],
    vram: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    max_tokens sensitivity: fixed-length workloads at several decode lengths.

    At a single fixed max_tokens the token feature barely varies, so (s, b) is
    ill-conditioned and OLS reports a degenerate fit. This cell quantifies that
    and justifies the mixed-length protocol used in D1/D2.
    """
    lengths = [int(x) for x in args.d3_lengths.split(",") if x.strip()]
    cells: Dict[str, Any] = {}
    for li, L in enumerate(lengths):
        rows = []
        for seed in range(args.d3_seeds):
            workload = build_workload(args.requests_per_seed, 7000 + seed, [L])
            row = _one_seed(
                session, backends, vram,
                strategy="nlms", seed=seed,
                port=args.dio_base_port + 600 + li * 20 + seed, args=args,
                workload=workload, tag=f"d3_{L}",
            )
            rows.append(row)
        good = [r for r in rows if not r.get("error")]
        slope_by_worker: Dict[str, Any] = {}
        for wid, _ in backends:
            vals = [
                (r.get("learned") or {}).get(wid, {}).get("fast_slope") for r in good
            ]
            slope_by_worker[wid] = mean_std([v for v in vals if isinstance(v, (int, float))])
        frac_acc: Dict[str, List[float]] = {}
        for r in good:
            for wid, f in (r.get("route_frac") or {}).items():
                frac_acc.setdefault(wid, []).append(f)
        cells[str(L)] = {
            "max_tokens": L,
            "p99": mean_std([r.get("e2e_p99_ms") for r in good]),
            "learned_slope": slope_by_worker,
            "route_frac": {k: statistics.mean(v) for k, v in frac_acc.items()},
            "ols_degenerate_seeds": sum(
                1 for r in good
                for v in (r.get("ols") or {}).values()
                if isinstance(v, dict) and v.get("note")
            ),
            "seeds_ok": len(good),
        }
        log(f"  d3 max_tokens={L}: slopes={ {k: (v.get('mean') if v else None) for k, v in slope_by_worker.items()} }")
    return {"cells": cells, "lengths": lengths}


if __name__ == "__main__":
    raise SystemExit(main())


