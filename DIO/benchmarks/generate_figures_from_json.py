#!/usr/bin/env python3
"""
Generate paper figures from benchmarks/results_summary.json (no hardcoded metrics).

Usage:
  python benchmarks/generate_figures_from_json.py
  python benchmarks/generate_figures_from_json.py --json benchmarks/results_summary.json --out figs
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["ShareGPT", "Arxiv", "Azure"]
STRATEGIES = ["NLMS", "RoundRobin"]
STRATEGY_LABELS = {"NLMS": "DIO (NLMS)", "RoundRobin": "Round Robin", "RLS": "RLS Baseline"}


def load_summary(path):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run analyze_results.py first.", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lookup(tests, strategy, dataset, workers=4):
    for t in tests:
        if t["strategy"] == strategy and t["dataset"] == dataset and t["workers"] == workers:
            return t
    return None


def plot_p99_comparison(tests, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DATASETS))
    width = 0.35
    dio_vals, rr_vals = [], []
    for ds in DATASETS:
        d = lookup(tests, "NLMS", ds)
        r = lookup(tests, "RoundRobin", ds)
        dio_vals.append(d["p99_s"] if d else 0)
        rr_vals.append(r["p99_s"] if r else 0)
    ax.bar(x - width / 2, dio_vals, width, label="DIO (NLMS)", color="#2ca02c")
    ax.bar(x + width / 2, rr_vals, width, label="Round Robin", color="#d62728")
    ax.set_ylabel("p99 Latency (s)")
    ax.set_title("Tail Latency (p99) — 4 Workers (from results_summary.json)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_6_line_comparison_clean.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")


def plot_slo_attainment(tests, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DATASETS))
    width = 0.35
    dio_vals, rr_vals = [], []
    for ds in DATASETS:
        d = lookup(tests, "NLMS", ds)
        r = lookup(tests, "RoundRobin", ds)
        dio_vals.append(d["slo_attainment_pct"] if d else 0)
        rr_vals.append(r["slo_attainment_pct"] if r else 0)
    ax.bar(x - width / 2, dio_vals, width, label="DIO (NLMS)", color="#2ca02c")
    ax.bar(x + width / 2, rr_vals, width, label="Round Robin", color="#d62728")
    ax.set_ylabel("SLO Attainment (%)")
    ax.set_title("SLO Attainment — 4 Workers")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_slo_from_json.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=os.path.join(root, "benchmarks", "results_summary.json"))
    parser.add_argument("--out", default=os.path.join(root, "..", "figs"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = load_summary(args.json)
    tests = summary["tests"]
    plot_p99_comparison(tests, args.out)
    plot_slo_attainment(tests, args.out)


if __name__ == "__main__":
    main()