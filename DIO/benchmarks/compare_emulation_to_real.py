#!/usr/bin/env python3
"""
Compare calibrated mock latency profiles against observed real-worker telemetry.

Usage (after T2 or a short probe run):
  python benchmarks/compare_emulation_to_real.py --real-log worker_0.log
  python benchmarks/compare_emulation_to_real.py --pairing t4_vs_a100 --samples 50

Writes benchmarks/emulation_validation.json for paper Table / threats-to-validity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from mock_latency_model import MockLatencySimulator, resolve_profile, load_profiles


def parse_latencies_from_log(path: str) -> list[float]:
    """Best-effort parse of latency_ms from worker or manager logs."""
    latencies = []
    patterns = [
        r"latency_ms[=:\s]+(\d+(?:\.\d+)?)",
        r"total[_\s]?latency[:\s]+(\d+(?:\.\d+)?)",
        r"Latency[:\s]+(\d+(?:\.\d+)?)\s*ms",
    ]
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            for pat in patterns:
                m = re.search(pat, line, re.I)
                if m:
                    latencies.append(float(m.group(1)))
                    break
    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairing", default="t4_vs_a100")
    parser.add_argument("--real-log", help="Log from real fast worker (optional)")
    parser.add_argument("--samples", type=int, default=40, help="Mock samples for distribution")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.output or os.path.join(root, "benchmarks", "emulation_validation.json")

    data = load_profiles()
    pairing = data["pairings"].get(args.pairing)
    if not pairing:
        print(f"Unknown pairing: {args.pairing}")
        sys.exit(1)

    fast_prof = resolve_profile(pairing_name=args.pairing, profile_role="fast")
    slow_prof = resolve_profile(pairing_name=args.pairing, profile_role="slow")
    slow_sim = MockLatencySimulator(slow_prof, seed=42)

    prompt = "Summarize the following paragraph in three bullet points."
    mock_latencies = [slow_sim.predict(prompt, 128).total_latency_ms for _ in range(args.samples)]

    result = {
        "pairing": args.pairing,
        "decode_slope_ratio_expected": pairing.get("decode_slope_ratio"),
        "decode_slope_ratio_observed": round(
            slow_prof.decode_slope_ms_per_token / fast_prof.decode_slope_ms_per_token, 2
        ),
        "mock_slow_ms": {
            "p50": round(sorted(mock_latencies)[len(mock_latencies) // 2], 1),
            "p99": round(sorted(mock_latencies)[int(len(mock_latencies) * 0.99)], 1),
            "mean": round(sum(mock_latencies) / len(mock_latencies), 1),
        },
        "sources": pairing.get("sources", []),
        "real_observed_ms": None,
        "ratio_real_to_mock_p50": None,
    }

    if args.real_log and os.path.exists(args.real_log):
        real = parse_latencies_from_log(args.real_log)
        if real:
            real_sorted = sorted(real)
            p50 = real_sorted[len(real_sorted) // 2]
            result["real_observed_ms"] = {
                "count": len(real),
                "p50": round(p50, 1),
                "p99": round(real_sorted[int(len(real_sorted) * 0.99)], 1),
            }
            result["ratio_real_to_mock_p50"] = round(
                result["mock_slow_ms"]["p50"] / p50, 2
            )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
    print("For paper: cite pairing sources + optional real-log validation ratio.")


if __name__ == "__main__":
    main()