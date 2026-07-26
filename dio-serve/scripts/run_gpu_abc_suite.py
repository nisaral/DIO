#!/usr/bin/env python3
"""
==============================================================================
REAL dual-T4 / dual-vLLM suite for title contributions A / B / C
==============================================================================
Closes the paper gap: hybrid, affinity, and admission on *real* engines, not only
the library multi-seed harness.

  G0  Attach/start dual engines (vLLM preferred)
  G1  Hybrid cost ON vs OFF vs RR under multi-turn long-prefix pressure
      (real /metrics scrape when engines export Prometheus)
  G2  Multi-turn session affinity: NLMS+affinity vs RR; DIO affinity hit rate
      + scraped engine prefix hit if available
  G3  Admission modes absolute vs empirical vs rank_only on the real gateway
      (tight SLO; measure rejects + absolute_vs_active_disagree)

Does NOT replace Regime A/C NLMS ranking tables — complements them.

Kaggle (after git pull)::

  !cd /kaggle/working/DIO/dio-serve && pip install -e . -q
  !python scripts/run_gpu_abc_suite.py \\
      --engine-mode vllm \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --gpus 0,1 \\
      --tokenizer Qwen/Qwen2.5-3B-Instruct \\
      --out /kaggle/working/results_gpu_abc

External engines already up::

  python scripts/run_gpu_abc_suite.py \\
      --backends http://127.0.0.1:8000,http://127.0.0.1:8001 \\
      --tokenizer Qwen/Qwen2.5-3B-Instruct \\
      --out results_gpu_abc

Quick smoke::

  python scripts/run_gpu_abc_suite.py --quick --backends ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import httpx
except ImportError:
    print("pip install httpx", file=sys.stderr)
    sys.exit(2)

# Reuse workshop process helpers
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_workshop_final_suite import (  # type: ignore
    PY,
    Session,
    kill_proc,
    log,
    mean_std,
    pct,
    start_dio,
    start_vllm,
    wait_url,
)

# Long shared system prefixes (agent-style) for affinity + KV pressure
SYSTEM_PREFIXES = [
    "SYSTEM: You are a careful research assistant. Always answer in short sentences. "
    "Context block: " + ("alpha beta gamma delta " * 40),
    "SYSTEM: You are a coding copilot for Python. Prefer clear code comments. "
    "Context block: " + ("import numpy as np; " * 30),
    "SYSTEM: You are a customer support agent for an API product. Be brief. "
    "Context block: " + ("ticket queue status policy " * 35),
    "SYSTEM: You are a tutor for linear algebra. One-sentence definitions only. "
    "Context block: " + ("vector matrix eigenvalue " * 40),
]

USER_TURNS = [
    "What is 2+2?",
    "Name one color.",
    "Say hello briefly.",
    "One benefit of sleep?",
    "Capital of France?",
]


def fetch_json(client: httpx.Client, url: str) -> Dict[str, Any]:
    try:
        r = client.get(url, timeout=5.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def scrape_backend_metrics(client: httpx.Client, backends: List[str]) -> Dict[str, Any]:
    """Best-effort raw /metrics parse via dio engine helper if available."""
    out: Dict[str, Any] = {}
    try:
        from dio.engine_metrics import parse_prometheus_text, snapshot_from_metrics
    except Exception:
        return {"error": "engine_metrics_import_failed"}

    for i, base in enumerate(backends):
        url = base.rstrip("/") + "/metrics"
        try:
            r = client.get(url, timeout=3.0)
            if r.status_code >= 400:
                out[f"e{i}"] = {"ok": False, "status": r.status_code}
                continue
            snap = snapshot_from_metrics(parse_prometheus_text(r.text))
            out[f"e{i}"] = snap.as_dict()
        except Exception as e:
            out[f"e{i}"] = {"ok": False, "error": type(e).__name__}
    return out


def start_dio_ext(
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
    engine_metrics: bool = True,
    cache_bonus_ms: float = 200.0,
    kv_cache_cost_ms: float = 800.0,
    engine_queue_cost_ms: float = 50.0,
) -> str:
    """Like workshop start_dio but with hybrid/affinity knobs via env + CLI."""
    cmd = [
        PY, "-m", "dio", "serve",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--strategy", strategy,
        "--nlms-mode", "dual",
        "--slo-ms", str(slo_ms),
        "--admission-mode", admission_mode,
        "--cache-bonus-ms", str(cache_bonus_ms),
    ]
    if admission_off:
        cmd.append("--admission-off")
    if engine_metrics:
        cmd.append("--engine-metrics")
    else:
        cmd.append("--no-engine-metrics")
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    for i, b in enumerate(backends):
        cmd.extend(["-b", f"e{i}={b}"])
    env = {
        "DIO_STRATEGY": strategy,
        "DIO_ADMISSION_MODE": admission_mode,
        "DIO_SLO_MS": str(slo_ms),
        "DIO_ADMISSION_OFF": "1" if admission_off else "0",
        "DIO_ENGINE_METRICS": "1" if engine_metrics else "0",
        "DIO_CACHE_BONUS_MS": str(cache_bonus_ms),
        "DIO_KV_CACHE_COST_MS": str(kv_cache_cost_ms),
        "DIO_ENGINE_QUEUE_COST_MS": str(engine_queue_cost_ms),
        "DIO_METRICS_INTERVAL_S": "0.5",
    }
    if tokenizer:
        env["DIO_TOKENIZER_NAME"] = tokenizer
        env["DIO_USE_TOKENIZER"] = "1"
    session.start(name, cmd, env=env)
    return f"http://127.0.0.1:{port}"


def run_multiturn_load(
    base: str,
    *,
    model: str,
    n_sessions: int,
    turns: int,
    max_tokens: int,
    seed: int,
    concurrent: int = 1,
) -> Dict[str, Any]:
    """
    Multi-turn sessions sharing a long system prefix per session.
    concurrent>1 issues overlapping requests to build real queue/KV pressure.
    """
    random.seed(seed)
    e2e: List[float] = []
    routes: Dict[str, int] = {}
    sticky = 0
    total_turns = 0
    session_home: Dict[int, str] = {}
    ok = fail = 0
    rejects = 0

    with httpx.Client(timeout=300.0) as client:
        # reset stats
        try:
            client.post(f"{base.rstrip('/')}/debug/reset_stats")
        except Exception:
            pass

        for s in range(n_sessions):
            sys_p = SYSTEM_PREFIXES[s % len(SYSTEM_PREFIXES)]
            for t in range(turns):
                user = USER_TURNS[t % len(USER_TURNS)] + f" (sess={s} turn={t} seed={seed})"
                messages = [
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user},
                ]
                t0 = time.perf_counter()
                try:
                    r = client.post(
                        f"{base.rstrip('/')}/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": 0.0,
                        },
                    )
                    ms = (time.perf_counter() - t0) * 1000.0
                    if r.status_code == 200:
                        ok += 1
                        e2e.append(ms)
                        wid = r.headers.get("X-DIO-Backend") or "unknown"
                        routes[wid] = routes.get(wid, 0) + 1
                        if s not in session_home:
                            session_home[s] = wid
                        if wid == session_home[s]:
                            sticky += 1
                        total_turns += 1
                    elif r.status_code == 503:
                        fail += 1
                        rejects += 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1

                # Optional: fire concurrent fillers for pressure (same prefix)
                if concurrent > 1 and t == 0:
                    for c in range(concurrent - 1):
                        try:
                            client.post(
                                f"{base.rstrip('/')}/v1/chat/completions",
                                json={
                                    "model": model,
                                    "messages": messages,
                                    "max_tokens": max(8, max_tokens // 2),
                                    "temperature": 0.0,
                                },
                                timeout=120.0,
                            )
                        except Exception:
                            pass

        dbg = fetch_json(client, f"{base.rstrip('/')}/debug/metrics")
        eng = fetch_json(client, f"{base.rstrip('/')}/debug/engine")
        aff = dbg.get("affinity") or eng.get("affinity") or {}

    total_r = sum(routes.values()) or 1
    return {
        "seed": seed,
        "ok": ok,
        "fail": fail,
        "rejects_503": rejects,
        "routes": routes,
        "frac_e0": routes.get("e0", 0) / total_r,
        "stickiness": sticky / max(1, total_turns),
        "e2e_p50_ms": pct(e2e, 50),
        "e2e_p99_ms": pct(e2e, 99),
        "e2e_mean_ms": statistics.mean(e2e) if e2e else None,
        "affinity": aff,
        "engine_debug": eng.get("snapshots") or dbg.get("engine_metrics"),
        "admission": (dbg.get("admission") or {}),
        "prediction": (dbg.get("prediction") or {}),
        "mape_pct": (dbg.get("prediction") or {}).get("mape_pct"),
    }


def run_admission_load(
    base: str,
    *,
    model: str,
    n: int,
    max_tokens: int,
    seed: int,
) -> Dict[str, Any]:
    """Sequential short prompts; count 503s and admission counters."""
    random.seed(seed)
    e2e: List[float] = []
    ok = fail = rejects = 0
    with httpx.Client(timeout=180.0) as client:
        try:
            client.post(f"{base.rstrip('/')}/debug/reset_stats")
        except Exception:
            pass
        for i in range(n):
            prompt = USER_TURNS[i % len(USER_TURNS)] + f" seed={seed} i={i}"
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
                )
                ms = (time.perf_counter() - t0) * 1000.0
                if r.status_code == 200:
                    ok += 1
                    e2e.append(ms)
                elif r.status_code == 503:
                    rejects += 1
                    fail += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
        dbg = fetch_json(client, f"{base.rstrip('/')}/debug/metrics")
    return {
        "seed": seed,
        "ok": ok,
        "fail": fail,
        "rejects_503": rejects,
        "reject_rate": rejects / max(1, n),
        "e2e_p99_ms": pct(e2e, 99),
        "e2e_mean_ms": statistics.mean(e2e) if e2e else None,
        "admission": dbg.get("admission") or {},
        "mape_pct": (dbg.get("prediction") or {}).get("mape_pct"),
    }


def multi_seed_block(
    session: Session,
    backends: List[str],
    *,
    configs: List[Dict[str, Any]],
    seeds: int,
    dio_base: int,
    model: str,
    tokenizer: str,
    runner: str,
    n_sessions: int,
    turns: int,
    max_tokens: int,
    concurrent: int,
    n_adm: int,
    label: str,
) -> Dict[str, Any]:
    """
    configs: list of dicts with strategy, engine_metrics, cache_bonus_ms,
             admission_mode, admission_off, slo_ms, name
    runner: multiturn | admission
    """
    log(f"=== {label} configs={[c.get('name') for c in configs]} seeds={seeds} ===")
    out: Dict[str, Any] = {"label": label, "seeds": seeds, "configs": {}}
    for ci, cfg in enumerate(configs):
        rows = []
        for seed in range(seeds):
            port = dio_base + ci * 50 + seed
            name = f"{label}_{cfg['name']}_s{seed}"
            url = start_dio_ext(
                session,
                backends=backends,
                port=port,
                strategy=cfg.get("strategy", "nlms"),
                name=name,
                tokenizer=tokenizer,
                admission_mode=cfg.get("admission_mode", "rank_only"),
                slo_ms=float(cfg.get("slo_ms", 180_000)),
                admission_off=bool(cfg.get("admission_off", True)),
                engine_metrics=bool(cfg.get("engine_metrics", True)),
                cache_bonus_ms=float(cfg.get("cache_bonus_ms", 200.0)),
                kv_cache_cost_ms=float(cfg.get("kv_cache_cost_ms", 800.0)),
                engine_queue_cost_ms=float(cfg.get("engine_queue_cost_ms", 50.0)),
            )
            if not wait_url(url + "/healthz", timeout=90):
                rows.append({"seed": seed, "error": "dio_start_failed"})
                if session.handles:
                    kill_proc(session.handles.pop())
                continue
            time.sleep(1.0)  # let metrics loop tick once
            if runner == "multiturn":
                row = run_multiturn_load(
                    url,
                    model=model,
                    n_sessions=n_sessions,
                    turns=turns,
                    max_tokens=max_tokens,
                    seed=1000 + seed,
                    concurrent=concurrent,
                )
            else:
                row = run_admission_load(
                    url,
                    model=model,
                    n=n_adm,
                    max_tokens=max_tokens,
                    seed=2000 + seed,
                )
            row["config"] = cfg["name"]
            rows.append(row)
            log(
                f"  {cfg['name']} seed={seed}: ok={row.get('ok')} fail={row.get('fail')} "
                f"p99={row.get('e2e_p99_ms')} stick={row.get('stickiness')} "
                f"rej={row.get('reject_rate', row.get('rejects_503'))} "
                f"aff={row.get('affinity')}"
            )
            if session.handles:
                kill_proc(session.handles.pop())
            time.sleep(0.4)

        p99s = [r["e2e_p99_ms"] for r in rows if r.get("e2e_p99_ms") is not None]
        sticks = [r["stickiness"] for r in rows if r.get("stickiness") is not None]
        rejs = [r["reject_rate"] for r in rows if r.get("reject_rate") is not None]
        aff_hits = []
        disagrees = []
        for r in rows:
            a = r.get("affinity") or {}
            if a.get("hit_rate") is not None:
                aff_hits.append(float(a["hit_rate"]))
            adm = r.get("admission") or {}
            if adm.get("absolute_vs_active_disagree") is not None:
                disagrees.append(float(adm["absolute_vs_active_disagree"]))
        block = {
            "e2e_p99": mean_std(p99s),
            "stickiness": mean_std(sticks) if sticks else mean_std([]),
            "reject_rate": mean_std(rejs) if rejs else mean_std([]),
            "affinity_hit_rate": mean_std(aff_hits) if aff_hits else mean_std([]),
            "abs_disagree": mean_std(disagrees) if disagrees else mean_std([]),
            "per_seed": rows,
            "config": cfg,
        }
        out["configs"][cfg["name"]] = block
        log(
            f"  >> {cfg['name']} p99={block['e2e_p99']} stick={block['stickiness']} "
            f"aff_hit={block['affinity_hit_rate']} rej={block['reject_rate']}"
        )
    return out


def write_snippets(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Real-GPU A/B/C suite snippets",
        "",
        f"Generated: {summary.get('generated_at')}",
        f"Backends: {summary.get('backends')}",
        "",
    ]
    g1 = summary.get("G1_hybrid") or {}
    if g1.get("configs"):
        lines.append("## G1 Hybrid (real multi-turn + metrics)")
        for name, b in g1["configs"].items():
            lines.append(
                f"- {name}: p99={b.get('e2e_p99')} stick={b.get('stickiness')}"
            )
        lines.append("")
    g2 = summary.get("G2_affinity") or {}
    if g2.get("configs"):
        lines.append("## G2 Affinity (real multi-turn)")
        for name, b in g2["configs"].items():
            lines.append(
                f"- {name}: stick={b.get('stickiness')} aff_hit={b.get('affinity_hit_rate')} "
                f"p99={b.get('e2e_p99')}"
            )
        lines.append("")
    g3 = summary.get("G3_admission") or {}
    if g3.get("configs"):
        lines.append("## G3 Admission (real gateway)")
        for name, b in g3["configs"].items():
            lines.append(
                f"- {name}: reject_rate={b.get('reject_rate')} abs_disagree={b.get('abs_disagree')} "
                f"ok_p99={b.get('e2e_p99')}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Real dual-vLLM suite for hybrid/affinity/admission")
    p.add_argument("--engine-mode", default="external", choices=["external", "vllm"])
    p.add_argument("--backends", default="", help="url0,url1 if external")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.88)
    p.add_argument("--out", default=str(ROOT / "results_gpu_abc"))
    p.add_argument("--seeds", type=int, default=5, help="multi-seed; use 3 for quick, 5–10 for paper")
    p.add_argument("--sessions", type=int, default=12)
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--concurrent", type=int, default=2, help="extra concurrent posts for KV/queue pressure")
    p.add_argument("--adm-n", type=int, default=40)
    p.add_argument("--adm-slo-ms", type=float, default=800.0, help="tight SLO to exercise admission")
    p.add_argument("--dio-base-port", type=int, default=9100)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-g1", action="store_true")
    p.add_argument("--skip-g2", action="store_true")
    p.add_argument("--skip-g3", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.seeds = min(args.seeds, 2)
        args.sessions = 6
        args.turns = 3
        args.adm_n = 20
        args.concurrent = 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = Session(out_dir)
    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "note": (
            "Real multi-instance engines. G1–G3 validate title mechanisms on GPU; "
            "library suites remain complementary multi-seed ablations."
        ),
    }

    try:
        # ---- engines ----
        if args.engine_mode == "external" or args.backends.strip():
            backends = [b.strip() for b in args.backends.split(",") if b.strip()]
            if len(backends) < 2:
                log("ERROR: need two --backends URLs")
                return 2
            with httpx.Client(timeout=5.0) as c:
                for b in backends:
                    try:
                        c.get(b.rstrip("/") + "/v1/models")
                    except Exception as e:
                        log(f"WARN backend {b}: {e}")
        else:
            gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
            if len(gpus) < 2:
                log("ERROR: need two GPUs")
                return 2
            ports = [8000, 8001]
            for g, port in zip(gpus[:2], ports):
                start_vllm(
                    session,
                    gpu=g,
                    port=port,
                    model=args.model,
                    max_model_len=args.max_model_len,
                    gpu_mem_util=args.gpu_mem_util,
                    name=f"vllm_{g}_{port}",
                )
            backends = [f"http://127.0.0.1:{p}" for p in ports]
            for b in backends:
                if not wait_url(b + "/v1/models", timeout=1200):
                    log(f"ERROR: engine not ready {b}")
                    return 3

        summary["backends"] = backends
        log(f"Backends: {backends}")

        # Probe whether /metrics exists
        with httpx.Client(timeout=5.0) as c:
            summary["engine_metrics_probe"] = scrape_backend_metrics(c, backends)
        log(f"metrics probe: {json.dumps(summary['engine_metrics_probe'])[:200]}")

        # ---- G1 Hybrid ----
        if not args.skip_g1:
            g1_cfgs = [
                {
                    "name": "nlms_hybrid_on",
                    "strategy": "nlms",
                    "engine_metrics": True,
                    "cache_bonus_ms": 200.0,
                    "admission_off": True,
                    "admission_mode": "rank_only",
                },
                {
                    "name": "nlms_hybrid_off",
                    "strategy": "nlms",
                    "engine_metrics": False,
                    "cache_bonus_ms": 0.0,
                    "admission_off": True,
                    "admission_mode": "rank_only",
                },
                {
                    "name": "rr",
                    "strategy": "round_robin",
                    "engine_metrics": False,
                    "cache_bonus_ms": 0.0,
                    "admission_off": True,
                    "admission_mode": "rank_only",
                },
            ]
            summary["G1_hybrid"] = multi_seed_block(
                session,
                backends,
                configs=g1_cfgs,
                seeds=args.seeds,
                dio_base=args.dio_base_port,
                model=args.model,
                tokenizer=args.tokenizer,
                runner="multiturn",
                n_sessions=args.sessions,
                turns=args.turns,
                max_tokens=args.max_tokens,
                concurrent=args.concurrent,
                n_adm=args.adm_n,
                label="G1",
            )
            # vs RR improvement
            g1 = summary["G1_hybrid"]["configs"]
            if "rr" in g1 and "nlms_hybrid_on" in g1:
                rr = g1["rr"]["per_seed"]
                hy = g1["nlms_hybrid_on"]["per_seed"]
                imps = []
                for i in range(min(len(rr), len(hy))):
                    a, b = rr[i].get("e2e_p99_ms"), hy[i].get("e2e_p99_ms")
                    if a and b and a > 0:
                        imps.append(100.0 * (a - b) / a)
                g1["nlms_hybrid_on"]["p99_vs_rr_pct"] = mean_std(imps)

        # ---- G2 Affinity ----
        if not args.skip_g2:
            g2_cfgs = [
                {
                    "name": "nlms_affinity",
                    "strategy": "nlms",
                    "engine_metrics": True,
                    "cache_bonus_ms": 400.0,
                    "admission_off": True,
                    "admission_mode": "rank_only",
                },
                {
                    "name": "rr",
                    "strategy": "round_robin",
                    "engine_metrics": True,
                    "cache_bonus_ms": 0.0,
                    "admission_off": True,
                    "admission_mode": "rank_only",
                },
            ]
            summary["G2_affinity"] = multi_seed_block(
                session,
                backends,
                configs=g2_cfgs,
                seeds=args.seeds,
                dio_base=args.dio_base_port + 200,
                model=args.model,
                tokenizer=args.tokenizer,
                runner="multiturn",
                n_sessions=args.sessions,
                turns=args.turns,
                max_tokens=args.max_tokens,
                concurrent=1,  # clean stickiness signal
                n_adm=args.adm_n,
                label="G2",
            )

        # ---- G3 Admission ----
        if not args.skip_g3:
            g3_cfgs = [
                {
                    "name": "absolute",
                    "strategy": "nlms",
                    "engine_metrics": True,
                    "cache_bonus_ms": 200.0,
                    "admission_off": False,
                    "admission_mode": "absolute",
                    "slo_ms": args.adm_slo_ms,
                },
                {
                    "name": "empirical",
                    "strategy": "nlms",
                    "engine_metrics": True,
                    "cache_bonus_ms": 200.0,
                    "admission_off": False,
                    "admission_mode": "empirical",
                    "slo_ms": args.adm_slo_ms,
                },
                {
                    "name": "rank_only",
                    "strategy": "nlms",
                    "engine_metrics": True,
                    "cache_bonus_ms": 200.0,
                    "admission_off": False,
                    "admission_mode": "rank_only",
                    "slo_ms": args.adm_slo_ms,
                },
            ]
            summary["G3_admission"] = multi_seed_block(
                session,
                backends,
                configs=g3_cfgs,
                seeds=args.seeds,
                dio_base=args.dio_base_port + 400,
                model=args.model,
                tokenizer=args.tokenizer,
                runner="admission",
                n_sessions=args.sessions,
                turns=args.turns,
                max_tokens=args.max_tokens,
                concurrent=1,
                n_adm=args.adm_n,
                label="G3",
            )

        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        write_snippets(summary, out_dir / "paper_snippets.md")
        log(f"Wrote {out_dir / 'summary.json'}")
        log(f"Wrote {out_dir / 'paper_snippets.md'}")
        return 0
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    except Exception:
        log("FATAL")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        session.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
