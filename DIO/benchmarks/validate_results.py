#!/usr/bin/env python3
"""
Post-run sanity check for benchmark admissibility.
Exit 0 = results look publishable; exit 1 = red flags found.

Usage:
  python benchmarks/validate_results.py
  python benchmarks/validate_results.py --json benchmarks/results_summary.json
"""

import argparse
import json
import os
import sys

# Thresholds for 3B model, 2 workers, 120s Locust run on A100
MIN_REQUESTS = 80          # below = underloaded or broken
MAX_P99_MS = 30000         # above = CPU thrash / OOM fight
MIN_RPS = 0.8              # below = system barely serving
MAX_FAIL_RATE = 5.0        # percent
MIN_NLMS_SHAREGPT_ADVANTAGE = 0.0  # NLMS p99 should be <= RR (or explain)


def load_summary(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_test(tests, strategy, dataset, workers=2):
    for t in tests:
        if t.get("strategy") == strategy and t.get("dataset") == dataset and t.get("workers") == workers:
            return t
    return None


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=os.path.join(root, "benchmarks", "results_summary.json"))
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"FAIL: {args.json} not found. Run analyze_results.py first.")
        sys.exit(1)

    summary = load_summary(args.json)
    tests = summary.get("tests", [])
    passed, failed, warnings = 0, 0, 0

    def ok(msg):
        nonlocal passed
        print(f"  PASS: {msg}")
        passed += 1

    def fail(msg):
        nonlocal failed
        print(f"  FAIL: {msg}")
        failed += 1

    def warn(msg):
        nonlocal warnings
        print(f"  WARN: {msg}")
        warnings += 1

    print("=" * 50)
    print("DIO Results Admissibility Check")
    print("=" * 50)
    print(f"Source: {args.json}")
    print(f"Tests found: {len(tests)}")
    print()

    if len(tests) < 3:
        fail(f"Only {len(tests)} tests — expected at least 3 (ShareGPT NLMS/RR minimum)")

    core = find_test(tests, "NLMS", "ShareGPT", 2) or find_test(tests, "NLMS", "ShareGPT", 4)
    rr = find_test(tests, "RoundRobin", "ShareGPT", 2) or find_test(tests, "RoundRobin", "ShareGPT", 4)

    workers = 2
    if core and core.get("workers"):
        workers = core["workers"]

    for t in tests:
        name = t.get("id", "?")
        rc = t.get("request_count", 0)
        p99 = t.get("p99_ms", 0)
        rps = t.get("rps", 0)
        fr = t.get("fail_rate_pct", 0)

        if rc < MIN_REQUESTS:
            fail(f"{name}: only {rc} requests (need >={MIN_REQUESTS}) — likely underloaded or mock")
        else:
            ok(f"{name}: {rc} requests")

        if p99 > MAX_P99_MS:
            fail(f"{name}: p99={p99/1000:.1f}s too high — VRAM thrash or CPU")
        elif p99 < 500 and rc > 100:
            warn(f"{name}: p99={p99}ms suspiciously low — verify not mock")
        else:
            ok(f"{name}: p99={p99/1000:.1f}s in range")

        if rps < MIN_RPS:
            fail(f"{name}: RPS={rps:.2f} too low")
        else:
            ok(f"{name}: RPS={rps:.2f}")

        if fr > MAX_FAIL_RATE:
            fail(f"{name}: fail rate {fr}%")
        else:
            ok(f"{name}: fail rate {fr}%")

    if core and rr:
        nlms_p99 = core["p99_ms"]
        rr_p99 = rr["p99_ms"]
        if nlms_p99 <= rr_p99:
            ok(f"ShareGPT: NLMS p99 ({nlms_p99/1000:.1f}s) <= RR ({rr_p99/1000:.1f}s)")
        else:
            warn(f"ShareGPT: NLMS p99 ({nlms_p99/1000:.1f}s) > RR ({rr_p99/1000:.1f}s) — still publishable with explanation")
        if core["rps"] >= rr["rps"]:
            ok(f"ShareGPT: NLMS RPS ({core['rps']:.2f}) >= RR ({rr['rps']:.2f})")
    else:
        fail("Missing ShareGPT NLMS or RoundRobin run")

    rls = find_test(tests, "RLS", "ShareGPT", workers)
    if rls:
        ok(f"RLS baseline present (p99={rls['p99_ms']/1000:.1f}s)")
    else:
        warn("RLS baseline missing — ablation incomplete")

    print()
    print("=" * 50)
    print(f"Summary: {passed} pass, {failed} fail, {warnings} warn")
    if failed > 0:
        print("VERDICT: NOT ADMISSIBLE — fix issues and re-run")
        sys.exit(1)
    if warnings > 0:
        print("VERDICT: ADMISSIBLE WITH CAVEATS — review warnings for paper text")
        sys.exit(0)
    print("VERDICT: ADMISSIBLE — safe to update paper")
    sys.exit(0)


if __name__ == "__main__":
    main()