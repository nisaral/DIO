#!/usr/bin/env python3
"""Reanalyse committed DIO result summaries with paired bootstrap statistics.

This script never reruns an engine. It consumes JSON summaries already committed
to the repository and emits a machine-readable report plus a Markdown table.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def paired_stats(a: Iterable[float], b: Iterable[float], *, seed: int = 20260824, draws: int = 100_000) -> dict[str, Any]:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    d = y - x  # positive means baseline is slower than treatment
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    boot = d[idx].mean(axis=1)
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    dz = float(np.mean(d) / sd) if sd > 0 else (math.inf if np.mean(d) > 0 else 0.0)
    wins = int(np.sum(d > 0))
    ties = int(np.sum(d == 0))
    return {
        "n": int(n),
        "treatment_mean": float(np.mean(x)),
        "baseline_mean": float(np.mean(y)),
        "mean_absolute_delta": float(np.mean(d)),
        "relative_improvement_pct": float(100.0 * np.mean((y - x) / y)),
        "bootstrap_ci95_delta": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "cohens_dz": dz,
        "wins": wins,
        "ties": ties,
        "win_rate": float(wins / n) if n else None,
        "paired_deltas": [float(v) for v in d],
    }


def per_seed(section: dict[str, Any], config: str, field: str = "e2e_p99_ms") -> list[float]:
    out = []
    for row in section["configs"][config].get("per_seed", []):
        v = finite(row.get(field))
        if v is not None:
            out.append(v)
    return out


def rows_values(rows: list[dict[str, Any]], field: str = "e2e_p99_ms") -> list[float]:
    return [v for row in rows if (v := finite(row.get(field))) is not None]


def load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_reanalysis")
    ap.add_argument("--draws", type=int, default=100_000)
    args = ap.parse_args()
    out = ROOT / "dio-serve" / args.out
    out.mkdir(parents=True, exist_ok=True)
    abc = load("dio-serve/results_gpu_abc_n10/summary.json")
    regime_d = load("dio-serve/results_regime_d/summary.json")
    comparisons: list[dict[str, Any]] = []

    def add(name: str, section: dict[str, Any], treatment: str, baseline: str, note: str = "") -> None:
        comparisons.append({"comparison": name, "note": note, **paired_stats(
            per_seed(section, treatment), per_seed(section, baseline), draws=args.draws
        )})

    add("G1 hybrid-on vs round-robin", abc["G1_hybrid"], "nlms_hybrid_on", "rr", "dual-T4 multi-turn")
    add("G1 hybrid-on vs hybrid-off", abc["G1_hybrid"], "nlms_hybrid_on", "nlms_hybrid_off", "joint hybrid/affinity arm")
    add("G2 affinity vs round-robin", abc["G2_affinity"], "nlms_affinity", "rr", "dual-T4 multi-turn")
    add("G2 load affinity vs round-robin", abc["G2b_affinity_under_load"], "nlms_affinity_load", "rr_load", "concurrent multi-turn")
    d_rows = regime_d["D1_strategy_head_to_head"]["per_seed"]
    comparisons.append({"comparison": "Regime D NLMS vs round-robin", "note": "real T4/A30", **paired_stats(
        rows_values(d_rows["nlms"]), rows_values(d_rows["round_robin"]), draws=args.draws
    )})
    comparisons.append({"comparison": "Regime D NLMS vs least-loaded", "note": "real T4/A30", **paired_stats(
        rows_values(d_rows["nlms"]), rows_values(d_rows["least_loaded"]), draws=args.draws
    )})

    report = {"generated_from": ["dio-serve/results_gpu_abc_n10/summary.json", "dio-serve/results_regime_d/summary.json"], "draws": args.draws, "comparisons": comparisons}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Committed-result reanalysis", "", "Paired bootstrap (100,000 resamples; fixed seed 20260824). Positive deltas mean the baseline p99 is higher.", "", "| Comparison | n | Mean Δ ms | 95% CI Δ ms | Improvement | Cohen dz | Wins |", "|---|---:|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        ci = c["bootstrap_ci95_delta"]
        lines.append(f"| {c['comparison']} | {c['n']} | {c['mean_absolute_delta']:.1f} | [{ci[0]:.1f}, {ci[1]:.1f}] | {c['relative_improvement_pct']:.1f}% | {c['cohens_dz']:.2f} | {c['wins']}/{c['n']} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "REPORT.md")


if __name__ == "__main__":
    main()
