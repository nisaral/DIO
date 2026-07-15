#!/usr/bin/env python3
"""Multi-seed T2-style heterogeneity stats for the paper (Table t2_multiseed)."""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

from dio import Scheduler

OUT = Path(__file__).resolve().parents[1] / "results_validation" / "t2_multiseed.json"


class Sim:
    def __init__(self, id: str, slope: float, bias: float, jitter: float = 0.1):
        self.id = id
        self.slope = slope
        self.bias = bias
        self.jitter = jitter

    def serve(self, tokens: int) -> float:
        noise = 1.0 + random.uniform(-self.jitter, self.jitter)
        return max(1.0, (self.bias + self.slope * tokens) * noise)


def run_seed(seed: int, strategy: str = "nlms", n: int = 200):
    random.seed(seed)
    s = Scheduler(strategy=strategy, dual=True, admission_off=True, slo_ms=1e9)
    backs = {
        "fast": Sim("fast", 8.0, 40.0, 0.08),
        "slow": Sim("slow", 22.0, 90.0, 0.1),
    }
    for b in backs.values():
        s.register(b.id)
    routes: dict[str, int] = {}
    lats: list[float] = []
    for i in range(n):
        tokens = 40 + (i % 30) * 2
        w, _ = s.pick("x" * (tokens * 4), tokens=tokens)
        y = backs[w].serve(tokens)
        s.feedback(w, y, tokens)
        routes[w] = routes.get(w, 0) + 1
        lats.append(y)
    lats.sort()
    p99 = lats[int(0.99 * (len(lats) - 1))]
    return routes.get("fast", 0) / n, p99


def msc(xs: list[float]):
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return {"mean": m, "std": sd, "ci95": ci}


def main():
    seeds = list(range(10))
    nlms_f, nlms_p, rr_f, rr_p, imp = [], [], [], [], []
    for seed in seeds:
        f, p = run_seed(seed, "nlms")
        nlms_f.append(f)
        nlms_p.append(p)
        f2, p2 = run_seed(seed, "round_robin")
        rr_f.append(f2)
        rr_p.append(p2)
        imp.append((p2 - p) / p2 * 100.0)
    out = {
        "n_seeds": 10,
        "n_req": 200,
        "nlms_frac_fast": msc(nlms_f),
        "rr_frac_fast": msc(rr_f),
        "nlms_p99": msc(nlms_p),
        "rr_p99": msc(rr_p),
        "p99_improvement_pct": msc(imp),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
