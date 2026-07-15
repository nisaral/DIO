# Dual-T4 Kaggle run — honest analysis

**Status:** `ok`  
**Hardware:** 2× Tesla T4 (15 GB), CUDA torch 2.11+cu130  
**Model:** Qwen/Qwen2.5-3B-Instruct via **stock vLLM**  
**DIO:** dio-serve 0.2.0, NLMS / RR / LL, 3 seeds × 30 req, max_tokens=32  

## Headline (use in paper)

| Strategy | e2e p50 (ms) | e2e p99 (ms) | Success |
|----------|--------------|--------------|---------|
| **NLMS** | 500 ± 14 | **898 ± 9** | 90/90 |
| Round-Robin | 504 ± 14 | 999 ± 135 | 90/90 |
| Least-Loaded | 530 ± 26 | 988 ± 88 | 90/90 |

- **~10% lower mean p99** vs RR (898 vs 999).  
- **Much more stable tails:** p99 std **9 ms** (NLMS) vs **135 ms** (RR).  
- **100% success** — real engines, not mocks.

## MAPE is bad — say so first

| Suite | MAPE (approx) |
|-------|----------------|
| G1 smoke | ~67% |
| G2 NLMS multi-seed | ~76–82% (often ~81%) |
| G4 dual-µ | ~81.6% |
| G4 single-µ | ~85.9% |

**Example under-prediction:** predicted 55–193 ms vs actual 194–1222 ms early in a seed.  
**Cause:** cold-start slopes too low vs fixed overhead + small-batch e2e; token feature ≈ prompt/4 + max_tokens; scalar linear model under continuous batching.

**Honest paper frame (do not claim accurate ms prediction):**

> Absolute MAPE on dual-T4 short-decode is ~66–95%. DIO’s contribution is **relative cost ranking** under fixed hardware. G7 / Regime C shows 97.5% traffic to the fast worker when slopes differ — ranking works even when absolute error is large.

## What is *not* a failure

### G3 “only 7.8% hetero improvement / 43% to e0” (unthrottled)
Both backends are **identical T4 + same model**. NLMS should **not** invent a 75/25 split.  
Seeing ~43% vs 50% means workers look similar online — correct behavior.  
Strong skew results stay in the **slope-skew simulation** (G7: 97.5% to fast, 38% p99 cut) and in the **throttled real-GPU G3** (one peer behind delay proxy ×2, prefer 5 seeds).

### G4 dual ≈ single p99
On short, stable decode (`max_tokens=32`) without induced burst/thermal, dual-µ and single-µ look similar. Dual still has slightly better MAPE (81.6 vs 85.9). Don’t overclaim dual-µ here; cite controlled burst/thermal microbench for that story.

### G5 admission never 503’d
Tight SLO was 3000 ms but measured e2e was ~1 s and predictors stayed under budget → all admitted. To stress admission, use lower `tight-slo-ms` (e.g. 400–800) or longer `max_tokens` / higher concurrency.

## Do **not** fabricate

We do **not** rewrite these numbers to 17%/41%/75% on dual-T4.  
That would be scientific fraud. Framing:

1. **Regime A — Homogeneous dual-GPU** → stable p99 (this run).  
2. **Regime B — Legacy Locust ShareGPT 67s/81s** → different setup (4 workers, long decode); never mix with Regime A ms.  
3. **Regime C — Slope-skew** → multi-seed sim + optional real delay-proxy hetero.

## Done: real dual-T4 hetero multi-seed (see `results_gpu_cluster_hetero/`)

| Metric | NLMS | RR |
|--------|------|-----|
| Frac to fast (e0) | **0.680 ± 0.073** (n=5) | 0.500 |
| p99 improvement | **34.3% ± 7.1%** | — |
| e2e p99 (ms) | 2058 ± 237 | 3196 ± 747 |

Delay proxy ×2 on e1; real vLLM tokens. Complements G7 sim (97.5% / 38.3%).

## Paper paste lines

```text
On dual Tesla T4 GPUs serving Qwen2.5-3B-Instruct with stock vLLM
(n=3 seeds × 30 requests), DIO-NLMS achieves end-to-end p99 latency of
898±9 ms versus 999±135 ms for Round-Robin (100% success). Absolute MAPE
is high (~66–95%); we do not claim point-accurate latency prediction.
On identical GPUs, routing remains roughly balanced and NLMS primarily
reduces tail variance. When worker costs differ, multi-seed slope-skew
simulations place 97.5% of requests on the faster worker and reduce p99
by 38.3%±2.0% relative to Round-Robin — evidence that relative ranking
is what drives routing quality.
```
