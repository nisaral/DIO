#!/usr/bin/env python3
"""Draw the Regime D figure from a completed run's summary.json.

Two panels, because Regime D makes two separate claims and they need separate
evidence:

  (a) Slope recovery -- the online NLMS slope next to an offline OLS fit on the
      same observed data, per GPU SKU. OLS is the ground truth here. The point
      is not that the numbers match (they will not; NLMS is a tracking filter,
      so its slopes are shrunk toward zero) but that the *ordering* survives,
      because ranking is all the scheduler consumes.

  (b) Tail latency by strategy, with the spread across seeds. A mean that looks
      good with error bars overlapping the baseline is not a result, and this
      panel is drawn so a reader can see that for themselves.

    python scripts/plot_regime_d.py \
        --summary results_regime_d/summary.json \
        --out ../paper_drafts_latex/cluster_computing_submission/fig_regime_d.png

Refuses to run on a --mock summary: fixture numbers must never reach the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless gateway node
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STRAT_LABEL = {
    "nlms": "NLMS",
    "rls": "RLS",
    "least_loaded": "Least-Loaded",
    "round_robin": "RR",
}
STRAT_ORDER = ("nlms", "rls", "least_loaded", "round_robin")

C_LEARNED = "#2ca02c"  # same green as DIO elsewhere in the paper
C_OLS = "#7f7f7f"      # grey: the offline reference, not a competing method
C_BASELINE = "#d62728"


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def panel_slopes(ax, d2: dict) -> None:
    by_worker = d2.get("slopes_by_worker") or {}
    if not by_worker:
        die("D2 cell has no slopes_by_worker -- re-run the harness with --d2")

    # Order slowest -> fastest by measured per-token cost, so the bars descend
    # left to right. Not by declared VRAM: two cards can carry the same 24 GB
    # and differ 3x in memory bandwidth, which is the axis this figure is about.
    def rank(w: str) -> tuple:
        o = (by_worker[w].get("ols") or {}).get("slope")
        l = (by_worker[w].get("learned") or {}).get("mean")
        s = o if isinstance(o, (int, float)) and o > 0 else l
        return (0, -s, w) if isinstance(s, (int, float)) else (1, 0.0, w)

    wids = sorted(by_worker, key=rank)

    learned = [(by_worker[w].get("learned") or {}).get("mean") or 0.0 for w in wids]
    lerr = [(by_worker[w].get("learned") or {}).get("std") or 0.0 for w in wids]
    ols = [(by_worker[w].get("ols") or {}).get("slope") or 0.0 for w in wids]

    x = np.arange(len(wids))
    width = 0.36
    ax.bar(x - width / 2, learned, width, yerr=lerr, capsize=3,
           label="NLMS (online)", color=C_LEARNED)
    ax.bar(x + width / 2, ols, width, label="OLS (offline fit)", color=C_OLS)

    # Flag any worker whose OLS fit was degenerate -- an unmarked bar there
    # would look like a measurement rather than an ill-conditioned fit.
    for i, w in enumerate(wids):
        if (by_worker[w].get("ols") or {}).get("note"):
            ax.text(i + width / 2, ols[i], "*", ha="center", va="bottom",
                    fontsize=13, color=C_BASELINE)

    ax.set_ylabel(r"per-token slope $s$ (ms/token)")
    ax.set_xticks(x)
    ax.set_xticklabels([w.upper() for w in wids])
    ax.set_title("(a) Learned vs offline slope per SKU", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def panel_tail(ax, d1: dict) -> None:
    strats = [s for s in STRAT_ORDER if isinstance(d1.get(s), dict)]
    if not strats:
        die("D1 cell has no strategy results")

    p99 = [(d1[s].get("p99") or {}).get("mean") or 0.0 for s in strats]
    err = [(d1[s].get("p99") or {}).get("std") or 0.0 for s in strats]
    # RR is the baseline everything is compared against, so colour it apart
    # from the learned strategies instead of giving each strategy its own hue.
    colors = [C_BASELINE if s == "round_robin" else C_LEARNED for s in strats]

    x = np.arange(len(strats))
    ax.bar(x, p99, 0.6, yerr=err, capsize=3, color=colors)
    ax.set_ylabel("p99 latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_LABEL[s] for s in strats], fontsize=9)
    ax.set_title("(b) Tail latency by strategy (mean $\\pm$ s.d. over seeds)",
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate the paired win count. The mean alone hides whether an
    # improvement is consistent or carried by a single lucky seed.
    for i, s in enumerate(strats):
        if s == "round_robin":
            continue
        key = ("p99_improvement_vs_rr" if s == "nlms"
               else f"p99_improvement_vs_rr_{s}")
        imp = d1.get(key)
        if not isinstance(imp, dict) or imp.get("mean") is None:
            continue
        txt = f"{imp['mean']:.1f}%"
        if imp.get("wins") is not None:
            txt += f"\n({imp['wins']}/{imp['of']})"
        ax.text(i, p99[i] + (err[i] or 0) * 1.05, txt, ha="center", va="bottom",
                fontsize=8)
    ax.margins(y=0.18)  # headroom so the annotations are not clipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.summary.exists():
        die(f"no such summary: {args.summary}")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("mock"):
        die("this summary.json came from a --mock run. Mock numbers must never "
            "reach the paper. Re-run on real GPUs first.")

    d1 = summary.get("D1_strategy_head_to_head") or {}
    d2 = summary.get("D2_slope_identifiability") or {}
    if not d2:
        die("no D2 cell -- the slope panel is the point of this figure; "
            "re-run the harness with --d2")

    # Two columns at the width the paper's other figures use.
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    panel_slopes(axes[0], d2)
    panel_tail(axes[1], d1)
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")
    print("add to the .tex near \\label{sec:regimeD}:")
    print(f"  \\includegraphics[width=\\linewidth]{{{args.out.name}}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
