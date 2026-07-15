#!/usr/bin/env python3
"""
==============================================================================
DIO WORKSHOP FINAL SUITE — one Kaggle cell for all outstanding re-runs
==============================================================================
Closes the remaining evidence gaps on **the same dual-T4** (no mixed SKU needed):

  W0  Env probe
  W1  Start / attach dual real engines (vLLM preferred)
  W2  MAPE with HF tokenizer vs heuristic (cheap)
  W3  Regime A multi-seed matrix n=10  (NLMS / RLS / RR / LL) + tokenizer
  W4  Regime C real throttle (delay-proxy ×2) n=10  NLMS / RLS / RR
  W5  Longer decode multi-seed (max_tokens=128) n=3
  W6  Coefficient ±50% sensitivity on live dual-T4 (tier/cache/vram scale)

Does **NOT** require a second GPU SKU. True T4+L4 remains optional / limitation #1.

Kaggle notebook (after git pull):
  !cd /kaggle/working/DIO/dio-serve && pip install -e . -q
  !python scripts/run_workshop_final_suite.py \\
      --engine-mode vllm \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --gpus 0,1 \\
      --tokenizer Qwen/Qwen2.5-3B-Instruct \\
      --out /kaggle/working/results_workshop_final

Quick smoke (fewer seeds):
  !python scripts/run_workshop_final_suite.py --quick --gpus 0,1 ...

Skip phases:
  --skip-w2 --skip-w3 --skip-w4 --skip-w5 --skip-w6
  --only w2,w3,w4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import httpx
except ImportError:
    print("pip install httpx", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DELAY_PROXY = ROOT / "scripts" / "latency_delay_proxy.py"
sys.path.insert(0, str(ROOT / "src"))

PROMPTS = [
    "What is 2+2? Answer in one short sentence.",
    "Name three colors. Be brief.",
    "Explain gravity in one sentence.",
    "Write a haiku about rain.",
    "List two benefits of exercise.",
    "What is the capital of France? One word if possible.",
    "Summarize photosynthesis in one sentence.",
    "Say hello and ask how I am.",
    "List three programming languages briefly.",
    "What causes seasons on Earth? One sentence.",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mean_std(xs: Sequence[float]) -> Dict[str, float]:
    xs = [float(x) for x in xs]
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


def kill_proc(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def wait_url(url: str, timeout: float = 900.0, interval: float = 2.0) -> bool:
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
        self._files: List[Any] = []

    def start(
        self,
        name: str,
        cmd: List[str],
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.Popen:
        log_path = self.logs / f"{name}.log"
        f = open(log_path, "w", encoding="utf-8")
        self._files.append(f)
        e = os.environ.copy()
        if env:
            e.update(env)
        kwargs: Dict[str, Any] = {
            "stdout": f,
            "stderr": subprocess.STDOUT,
            "cwd": str(ROOT),
            "env": e,
        }
        if os.name != "nt":
            kwargs["preexec_fn"] = os.setsid
        log(f"START {name}: {' '.join(cmd[:14])}{'...' if len(cmd) > 14 else ''}")
        p = subprocess.Popen(cmd, **kwargs)
        self.handles.append(p)
        return p

    def cleanup(self) -> None:
        log("Cleaning up processes...")
        for p in reversed(self.handles):
            kill_proc(p)
        self.handles.clear()
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()
        time.sleep(1.0)


def ensure_delay_proxy() -> None:
    if DELAY_PROXY.exists():
        return
    # minimal writer — prefer repo file
    raise FileNotFoundError(
        f"Missing {DELAY_PROXY}. git pull feat-v2-rls-scheduler so latency_delay_proxy.py exists."
    )


def start_vllm(
    session: Session,
    *,
    gpu: str,
    port: int,
    model: str,
    max_model_len: int,
    gpu_mem_util: float,
    name: str,
) -> None:
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
    ]
    session.start(name, cmd, env={"CUDA_VISIBLE_DEVICES": str(gpu)})


def start_dio(
    session: Session,
    *,
    backends: List[str],
    port: int,
    strategy: str,
    name: str,
    tokenizer: str = "",
    admission_mode: str = "rank_only",
    slo_ms: float = 180_000,
    admission_off: bool = True,
    extra_env: Optional[Dict[str, str]] = None,
) -> str:
    cmd = [
        PY, "-m", "dio", "serve",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--strategy", strategy,
        "--nlms-mode", "dual",
        "--slo-ms", str(slo_ms),
        "--admission-mode", admission_mode,
    ]
    if admission_off:
        cmd.append("--admission-off")
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    for i, b in enumerate(backends):
        cmd.extend(["-b", f"e{i}={b}"])
    env = {
        "DIO_STRATEGY": strategy,
        "DIO_ADMISSION_MODE": admission_mode,
        "DIO_SLO_MS": str(slo_ms),
        "DIO_ADMISSION_OFF": "1" if admission_off else "0",
    }
    if tokenizer:
        env["DIO_TOKENIZER_NAME"] = tokenizer
        env["DIO_USE_TOKENIZER"] = "1"
    if extra_env:
        env.update(extra_env)
    session.start(name, cmd, env=env)
    return f"http://127.0.0.1:{port}"


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
    routes: Dict[str, int] = {}
    ok = fail = 0
    with httpx.Client(timeout=300.0) as client:
        try:
            h = client.get(f"{base.rstrip('/')}/healthz")
            health = h.json() if h.status_code == 200 else {"status": h.status_code}
        except Exception as e:
            return {"error": str(e), "ok": 0, "fail": n, "seed": seed}

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

        pred: Dict[str, Any] = {}
        try:
            m = client.get(f"{base.rstrip('/')}/debug/metrics").json()
            mapes = []
            for w in (m.get("workers") or {}).values():
                if isinstance(w, dict) and w.get("mape_pct") is not None:
                    mapes.append(float(w["mape_pct"]))
            pred = {
                "mape_pct": statistics.mean(mapes) if mapes else None,
                "prediction_block": m.get("prediction"),
                "admission": m.get("admission"),
            }
            if m.get("prediction") and m["prediction"].get("mape_pct") is not None:
                pred["mape_pct"] = m["prediction"]["mape_pct"]
        except Exception as e:
            pred = {"error": str(e)}

    total = sum(routes.values()) or 1
    return {
        "seed": seed,
        "n": n,
        "ok": ok,
        "fail": fail,
        "routes": routes,
        "frac_e0": routes.get("e0", 0) / total,
        "e2e_p50_ms": pct(e2e, 50),
        "e2e_p99_ms": pct(e2e, 99),
        "e2e_mean_ms": statistics.mean(e2e) if e2e else None,
        "mape_pct": pred.get("mape_pct"),
        "prediction": pred,
        "health": health,
    }


def multi_seed_matrix(
    session: Session,
    backends: List[str],
    *,
    strategies: List[str],
    seeds: int,
    n_per_seed: int,
    max_tokens: int,
    model: str,
    tokenizer: str,
    dio_base: int,
    label: str,
) -> Dict[str, Any]:
    log(f"=== {label}: strategies={strategies} seeds={seeds} n={n_per_seed} max_tokens={max_tokens} ===")
    out: Dict[str, Any] = {
        "label": label,
        "seeds": seeds,
        "n_per_seed": n_per_seed,
        "max_tokens": max_tokens,
        "tokenizer": tokenizer or "heuristic",
        "strategies": {},
    }
    for si, strat in enumerate(strategies):
        rows = []
        for seed in range(seeds):
            port = dio_base + si * 40 + seed
            url = start_dio(
                session,
                backends=backends,
                port=port,
                strategy=strat,
                name=f"{label}_{strat}_s{seed}",
                tokenizer=tokenizer,
                admission_mode="rank_only",
                admission_off=True,
            )
            if not wait_url(url + "/healthz", timeout=90):
                rows.append({"seed": seed, "error": "dio_start_failed"})
                if session.handles:
                    kill_proc(session.handles.pop())
                continue
            row = run_load(url, model=model, n=n_per_seed, max_tokens=max_tokens, seed=1000 * (si + 1) + seed)
            row["strategy"] = strat
            rows.append(row)
            log(
                f"  {strat} seed={seed}: ok={row.get('ok')}/{row.get('n')} "
                f"p99={row.get('e2e_p99_ms')} mape={row.get('mape_pct')} "
                f"frac_e0={row.get('frac_e0')} routes={row.get('routes')}"
            )
            if session.handles:
                kill_proc(session.handles.pop())
            time.sleep(0.5)

        p99s = [r["e2e_p99_ms"] for r in rows if r.get("e2e_p99_ms") is not None]
        p50s = [r["e2e_p50_ms"] for r in rows if r.get("e2e_p50_ms") is not None]
        mapes = [r["mape_pct"] for r in rows if r.get("mape_pct") is not None]
        fracs = [r["frac_e0"] for r in rows if r.get("frac_e0") is not None]
        out["strategies"][strat] = {
            "e2e_p99": mean_std(p99s),
            "e2e_p50": mean_std(p50s),
            "mape": mean_std(mapes),
            "frac_e0": mean_std(fracs),
            "ok_sum": sum(r.get("ok", 0) for r in rows),
            "per_seed": rows,
        }
        m = out["strategies"][strat]["e2e_p99"]
        log(f"  >> {strat} p99 {m['mean']:.1f}±{m['std']:.1f} (n={m['n']}) mape={out['strategies'][strat]['mape']}")
    # vs RR
    rr = out["strategies"].get("round_robin") or out["strategies"].get("rr")
    if rr and rr.get("per_seed"):
        rr_p99 = [r["e2e_p99_ms"] for r in rr["per_seed"] if r.get("e2e_p99_ms")]
        for strat, block in out["strategies"].items():
            if strat in ("round_robin", "rr"):
                continue
            imps = []
            for i, r in enumerate(block.get("per_seed") or []):
                if i < len(rr_p99) and r.get("e2e_p99_ms") and rr_p99[i] > 0:
                    imps.append((rr_p99[i] - r["e2e_p99_ms"]) / rr_p99[i] * 100.0)
            block["p99_improvement_vs_rr_pct"] = mean_std(imps)
    return out


def coeff_sweep_live(
    session: Session,
    backends: List[str],
    *,
    model: str,
    tokenizer: str,
    n: int,
    max_tokens: int,
    dio_base: int,
) -> Dict[str, Any]:
    """
    ±50% on tier / cache / vram soft scale via env (pydantic DIO_*).
    Defaults: tier=500, cache=200, vram soft uses internal 1000 scale — we scale
    DIO_TIER_MISMATCH_MS and DIO_CACHE_BONUS_MS (documented paper coefficients).
    """
    log("=== W6 Coefficient ±50% sensitivity (live dual backends) ===")
    variants = {
        "baseline": {"DIO_TIER_MISMATCH_MS": "500", "DIO_CACHE_BONUS_MS": "200"},
        "tier_p50": {"DIO_TIER_MISMATCH_MS": "750", "DIO_CACHE_BONUS_MS": "200"},
        "tier_m50": {"DIO_TIER_MISMATCH_MS": "250", "DIO_CACHE_BONUS_MS": "200"},
        "cache_p50": {"DIO_TIER_MISMATCH_MS": "500", "DIO_CACHE_BONUS_MS": "300"},
        "cache_m50": {"DIO_TIER_MISMATCH_MS": "500", "DIO_CACHE_BONUS_MS": "100"},
        "both_p50": {"DIO_TIER_MISMATCH_MS": "750", "DIO_CACHE_BONUS_MS": "300"},
        "both_m50": {"DIO_TIER_MISMATCH_MS": "250", "DIO_CACHE_BONUS_MS": "100"},
    }
    out: Dict[str, Any] = {"variants": {}, "n": n, "max_tokens": max_tokens}
    for i, (name, env) in enumerate(variants.items()):
        port = dio_base + i
        url = start_dio(
            session,
            backends=backends,
            port=port,
            strategy="nlms",
            name=f"coeff_{name}",
            tokenizer=tokenizer,
            admission_mode="rank_only",
            admission_off=True,
            extra_env=env,
        )
        if not wait_url(url + "/healthz", timeout=90):
            out["variants"][name] = {"error": "start_failed"}
            if session.handles:
                kill_proc(session.handles.pop())
            continue
        row = run_load(url, model=model, n=n, max_tokens=max_tokens, seed=7000 + i)
        out["variants"][name] = {
            "env": env,
            "e2e_p99_ms": row.get("e2e_p99_ms"),
            "e2e_p50_ms": row.get("e2e_p50_ms"),
            "mape_pct": row.get("mape_pct"),
            "frac_e0": row.get("frac_e0"),
            "ok": row.get("ok"),
        }
        log(f"  {name}: p99={row.get('e2e_p99_ms')} mape={row.get('mape_pct')} ok={row.get('ok')}")
        if session.handles:
            kill_proc(session.handles.pop())
        time.sleep(0.4)

    base_p99 = (out["variants"].get("baseline") or {}).get("e2e_p99_ms")
    if base_p99 and base_p99 > 0:
        deltas = {}
        for name, block in out["variants"].items():
            if name == "baseline" or block.get("e2e_p99_ms") is None:
                continue
            deltas[name] = (block["e2e_p99_ms"] - base_p99) / base_p99 * 100.0
        out["p99_delta_pct_vs_baseline"] = deltas
        if deltas:
            out["max_abs_delta_pct"] = max(abs(v) for v in deltas.values())
            log(f"  >> max |Δp99| vs baseline = {out['max_abs_delta_pct']:.2f}%")
    return out


def write_report(summary: Dict[str, Any], out: Path) -> None:
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Workshop final suite — paper snippets",
        f"Generated: {summary.get('generated_at')}",
        f"Model: {summary.get('args', {}).get('model')}",
        f"Tokenizer: {summary.get('args', {}).get('tokenizer')}",
        f"Status: {summary.get('status')}",
        "",
    ]

    def dump_matrix(title: str, key: str) -> None:
        block = summary.get(key)
        if not block or block.get("skipped") or block.get("error"):
            return
        lines.append(f"## {title}")
        for strat, st in (block.get("strategies") or {}).items():
            p = st.get("e2e_p99") or {}
            m = st.get("mape") or {}
            f = st.get("frac_e0") or {}
            imp = st.get("p99_improvement_vs_rr_pct") or {}
            lines.append(
                f"- **{strat}**: p99 ${p.get('mean', float('nan')):.1f}\\pm{p.get('std', 0):.1f}$ "
                f"(ci95 $\\pm{p.get('ci95', 0):.1f}$, n={p.get('n')}); "
                f"MAPE ${m.get('mean', float('nan')):.1f}\\pm{m.get('std', 0):.1f}$; "
                f"frac_e0 ${f.get('mean', float('nan')):.3f}\\pm{f.get('std', 0):.3f}$"
                + (
                    f"; vs RR ${imp.get('mean', float('nan')):.1f}\\pm{imp.get('std', 0):.1f}\\%$"
                    if imp.get("n")
                    else ""
                )
            )
        lines.append("")

    w2 = summary.get("W2_mape_tokenizer")
    if w2 and not w2.get("skipped"):
        lines.append("## W2 MAPE tokenizer vs heuristic")
        for k in ("heuristic", "tokenizer"):
            b = w2.get(k) or {}
            m = b.get("mape") or {}
            lines.append(
                f"- **{k}**: MAPE ${m.get('mean', float('nan')):.1f}\\pm{m.get('std', 0):.1f}$ (n={m.get('n')})"
            )
        if w2.get("mape_delta_pp") is not None:
            lines.append(f"- delta (tok − heur) pp: {w2['mape_delta_pp']:.2f}")
        lines.append("")

    dump_matrix("W3 Regime A n=10 (homogeneous dual-GPU)", "W3_regime_a")
    dump_matrix("W4 Regime C real throttle n=10", "W4_regime_c_throttle")
    dump_matrix("W5 Longer decode max_tokens=128", "W5_long_decode")

    w6 = summary.get("W6_coeff_sweep")
    if w6 and not w6.get("skipped"):
        lines.append("## W6 Coefficient ±50% (live)")
        lines.append(f"- max |Δp99| vs baseline: {w6.get('max_abs_delta_pct')}")
        for name, d in (w6.get("p99_delta_pct_vs_baseline") or {}).items():
            lines.append(f"- {name}: {d:.2f}%")
        lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- Full JSON: `{out / 'summary.json'}`")
    lines.append(f"- Logs: `{out / 'logs'}`")
    (out / "paper_snippets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote {out / 'paper_snippets.md'}")


def parse_args():
    p = argparse.ArgumentParser(description="DIO workshop final suite (all remaining re-runs)")
    p.add_argument("--out", default=str(ROOT / "results_workshop_final"))
    p.add_argument("--engine-mode", choices=["vllm", "external"], default="vllm")
    p.add_argument("--backends", default="", help="external: url0,url1")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--engine-base-port", type=int, default=18000)
    p.add_argument("--dio-base-port", type=int, default=19500)
    p.add_argument("--proxy-port", type=int, default=18101)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--slow-mult", type=float, default=2.0)

    # seed / load defaults (paper gate)
    p.add_argument("--seeds-a", type=int, default=10, help="Regime A seeds")
    p.add_argument("--seeds-c", type=int, default=10, help="Regime C throttle seeds")
    p.add_argument("--seeds-long", type=int, default=3, help="Long-decode seeds")
    p.add_argument("--seeds-mape", type=int, default=3, help="MAPE tokenizer comparison seeds")
    p.add_argument("--n-per-seed", type=int, default=30)
    p.add_argument("--n-long", type=int, default=20, help="Requests/seed for long decode")
    p.add_argument("--n-coeff", type=int, default=24, help="Requests per coeff variant")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-tokens-long", type=int, default=128)

    p.add_argument("--strategies-a", default="nlms,rls,round_robin,least_loaded")
    p.add_argument("--strategies-c", default="nlms,rls,round_robin")
    p.add_argument("--strategies-long", default="nlms,round_robin")

    p.add_argument("--skip-w2", action="store_true")
    p.add_argument("--skip-w3", action="store_true")
    p.add_argument("--skip-w4", action="store_true")
    p.add_argument("--skip-w5", action="store_true")
    p.add_argument("--skip-w6", action="store_true")
    p.add_argument(
        "--only",
        default="",
        help="Comma list: w2,w3,w4,w5,w6 (overrides skips)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fewer seeds/requests for a short Kaggle smoke",
    )
    args, _ = p.parse_known_args()
    return args


def apply_only_flags(args) -> None:
    if args.only.strip():
        want = {x.strip().lower() for x in args.only.split(",") if x.strip()}
        args.skip_w2 = "w2" not in want
        args.skip_w3 = "w3" not in want
        args.skip_w4 = "w4" not in want
        args.skip_w5 = "w5" not in want
        args.skip_w6 = "w6" not in want
    if args.quick:
        args.seeds_a = min(args.seeds_a, 2)
        args.seeds_c = min(args.seeds_c, 2)
        args.seeds_long = min(args.seeds_long, 2)
        args.seeds_mape = min(args.seeds_mape, 2)
        args.n_per_seed = min(args.n_per_seed, 12)
        args.n_long = min(args.n_long, 8)
        args.n_coeff = min(args.n_coeff, 10)
        args.max_tokens_long = min(args.max_tokens_long, 64)


def save_partial(summary: Dict[str, Any], out: Path) -> None:
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    apply_only_flags(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = Session(out)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "run_workshop_final_suite.py",
        "args": vars(args),
        "purpose": [
            "W2 MAPE tokenizer vs heuristic",
            "W3 Regime A multi-seed n=10 NLMS/RLS/RR/LL",
            "W4 Regime C delay-proxy throttle n=10",
            "W5 longer decode max_tokens multi-seed",
            "W6 coefficient ±50% on dual-T4",
        ],
        "note": "Same dual-T4 only — no mixed SKU required",
    }

    log("=" * 64)
    log("DIO WORKSHOP FINAL SUITE")
    log("=" * 64)
    log(f"out={out}")
    log(
        f"seeds A={args.seeds_a} C={args.seeds_c} long={args.seeds_long} "
        f"tokenizer={args.tokenizer or 'heuristic'}"
    )

    backends: List[str] = []
    try:
        # ----- engines -----
        if args.engine_mode == "external" or args.backends.strip():
            backends = [b.strip() for b in args.backends.split(",") if b.strip()]
            if len(backends) < 2:
                log("ERROR: need two backends")
                return 2
            for b in backends:
                wait_url(b.rstrip("/") + "/v1/models", timeout=60)
            log(f"External backends: {backends}")
        else:
            gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
            if len(gpus) < 2:
                log("ERROR: need --gpus 0,1 for dual-T4 suite")
                return 2
            for i, gpu in enumerate(gpus[:2]):
                port = args.engine_base_port + i
                start_vllm(
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
                log(f"Waiting for {b} (vLLM may take several minutes)...")
                if not wait_url(b + "/v1/models", timeout=900):
                    log(f"ERROR: engine failed {b} — see logs/")
                    for h in session.handles:
                        # not tracking log path on handles simply
                        pass
                    summary["status"] = "engine_start_failed"
                    save_partial(summary, out)
                    return 1
        summary["backends"] = backends

        # smoke
        port = args.dio_base_port
        url = start_dio(
            session,
            backends=backends[:1],
            port=port,
            strategy="nlms",
            name="smoke",
            tokenizer=args.tokenizer,
        )
        if not wait_url(url + "/healthz", timeout=90):
            log("ERROR: DIO smoke failed")
            summary["status"] = "dio_smoke_failed"
            save_partial(summary, out)
            return 1
        smoke = run_load(url, model=args.model, n=2, max_tokens=args.max_tokens, seed=0)
        summary["W0_smoke"] = smoke
        log(f"Smoke ok={smoke.get('ok')} p99={smoke.get('e2e_p99_ms')}")
        if session.handles:
            kill_proc(session.handles.pop())
        if smoke.get("ok", 0) < 1:
            summary["status"] = "smoke_no_success"
            save_partial(summary, out)
            return 1

        # ----- W2 MAPE tokenizer vs heuristic -----
        if not args.skip_w2:
            log("=== W2 MAPE: heuristic vs HF tokenizer ===")
            w2: Dict[str, Any] = {}
            for tag, tok in (("heuristic", ""), ("tokenizer", args.tokenizer)):
                rows = []
                for seed in range(args.seeds_mape):
                    port = args.dio_base_port + 50 + (0 if tag == "heuristic" else 20) + seed
                    url = start_dio(
                        session,
                        backends=backends,
                        port=port,
                        strategy="nlms",
                        name=f"mape_{tag}_s{seed}",
                        tokenizer=tok,
                    )
                    if not wait_url(url + "/healthz", timeout=90):
                        rows.append({"seed": seed, "error": "start_failed"})
                        if session.handles:
                            kill_proc(session.handles.pop())
                        continue
                    row = run_load(
                        url,
                        model=args.model,
                        n=args.n_per_seed,
                        max_tokens=args.max_tokens,
                        seed=200 + seed,
                    )
                    rows.append(row)
                    log(f"  {tag} seed={seed}: mape={row.get('mape_pct')} p99={row.get('e2e_p99_ms')}")
                    if session.handles:
                        kill_proc(session.handles.pop())
                    time.sleep(0.4)
                mapes = [r["mape_pct"] for r in rows if r.get("mape_pct") is not None]
                p99s = [r["e2e_p99_ms"] for r in rows if r.get("e2e_p99_ms") is not None]
                w2[tag] = {
                    "mape": mean_std(mapes),
                    "e2e_p99": mean_std(p99s),
                    "per_seed": rows,
                    "tokenizer": tok or "heuristic",
                }
            # delta
            if w2.get("heuristic", {}).get("mape", {}).get("n") and w2.get("tokenizer", {}).get("mape", {}).get("n"):
                w2["mape_delta_pp"] = (
                    w2["tokenizer"]["mape"]["mean"] - w2["heuristic"]["mape"]["mean"]
                )
                log(
                    f"  >> MAPE heuristic={w2['heuristic']['mape']['mean']:.1f} "
                    f"tokenizer={w2['tokenizer']['mape']['mean']:.1f} "
                    f"delta={w2['mape_delta_pp']:.1f} pp"
                )
            summary["W2_mape_tokenizer"] = w2
            save_partial(summary, out)
        else:
            summary["W2_mape_tokenizer"] = {"skipped": True}

        # ----- W3 Regime A n=10 -----
        if not args.skip_w3:
            strats = [s.strip() for s in args.strategies_a.split(",") if s.strip()]
            summary["W3_regime_a"] = multi_seed_matrix(
                session,
                backends,
                strategies=strats,
                seeds=args.seeds_a,
                n_per_seed=args.n_per_seed,
                max_tokens=args.max_tokens,
                model=args.model,
                tokenizer=args.tokenizer,
                dio_base=args.dio_base_port + 200,
                label="W3_regime_a",
            )
            save_partial(summary, out)
        else:
            summary["W3_regime_a"] = {"skipped": True}

        # ----- W4 Regime C throttle -----
        if not args.skip_w4:
            ensure_delay_proxy()
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
                summary["W4_regime_c_throttle"] = {"error": "proxy_failed"}
            else:
                hetero = [backends[0], proxy_url]
                strats = [s.strip() for s in args.strategies_c.split(",") if s.strip()]
                summary["W4_regime_c_throttle"] = multi_seed_matrix(
                    session,
                    hetero,
                    strategies=strats,
                    seeds=args.seeds_c,
                    n_per_seed=args.n_per_seed,
                    max_tokens=args.max_tokens,
                    model=args.model,
                    tokenizer=args.tokenizer,
                    dio_base=args.dio_base_port + 400,
                    label="W4_regime_c",
                )
                summary["W4_regime_c_throttle"]["setup"] = {
                    "slow_mult": args.slow_mult,
                    "note": f"e1={raw_e1} behind delay proxy ×{args.slow_mult}",
                }
            save_partial(summary, out)
        else:
            summary["W4_regime_c_throttle"] = {"skipped": True}

        # ----- W5 long decode -----
        if not args.skip_w5:
            strats = [s.strip() for s in args.strategies_long.split(",") if s.strip()]
            summary["W5_long_decode"] = multi_seed_matrix(
                session,
                backends,  # homogeneous; no proxy required
                strategies=strats,
                seeds=args.seeds_long,
                n_per_seed=args.n_long,
                max_tokens=args.max_tokens_long,
                model=args.model,
                tokenizer=args.tokenizer,
                dio_base=args.dio_base_port + 600,
                label="W5_long_decode",
            )
            save_partial(summary, out)
        else:
            summary["W5_long_decode"] = {"skipped": True}

        # ----- W6 coefficient sweep -----
        if not args.skip_w6:
            summary["W6_coeff_sweep"] = coeff_sweep_live(
                session,
                backends,
                model=args.model,
                tokenizer=args.tokenizer,
                n=args.n_coeff,
                max_tokens=args.max_tokens,
                dio_base=args.dio_base_port + 700,
            )
            save_partial(summary, out)
        else:
            summary["W6_coeff_sweep"] = {"skipped": True}

        summary["status"] = "ok"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        log("Interrupted — partial results saved")
    except Exception as e:
        summary["status"] = "error"
        summary["error"] = str(e)
        log(f"FATAL: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.cleanup()

    write_report(summary, out)
    print("\n" + "=" * 64)
    print("WORKSHOP FINAL SUITE COMPLETE")
    print("=" * 64)
    print(f"Status: {summary.get('status')}")
    print(f"Results → {out}")
    print("  summary.json")
    print("  paper_snippets.md")
    print("  logs/")
    # quick headline
    for key in ("W3_regime_a", "W4_regime_c_throttle", "W5_long_decode"):
        b = summary.get(key) or {}
        st = (b.get("strategies") or {}).get("nlms") or {}
        p = st.get("e2e_p99") or {}
        if p.get("n"):
            print(f"  {key} NLMS p99: {p['mean']:.1f}±{p['std']:.1f}")
    w2 = summary.get("W2_mape_tokenizer") or {}
    if w2.get("tokenizer"):
        print(
            f"  W2 MAPE tok={ (w2.get('tokenizer') or {}).get('mape', {}).get('mean') } "
            f"heur={ (w2.get('heuristic') or {}).get('mape', {}).get('mean') }"
        )
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
