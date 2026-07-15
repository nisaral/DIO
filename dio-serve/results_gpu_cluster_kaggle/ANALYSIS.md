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

## What is *not* a failure

### G3 “only 7.8% hetero improvement / 43% to e0”
Both backends are **identical T4 + same model**. NLMS should **not** invent a 75/25 split.  
Seeing ~43% vs 50% means workers look similar online — correct behavior.  
Strong skew results stay in the **slope-skew simulation** (G7 / Table t2_multiseed: 97.5% to fast, 38% p99 cut).

### G4 dual ≈ single p99
On short, stable decode (`max_tokens=32`) without induced burst/thermal, dual-µ and single-µ look similar. Dual still has slightly better MAPE (81.6 vs 85.9). Don’t overclaim dual-µ here; cite controlled burst/thermal microbench for that story.

### G5 admission never 503’d
Tight SLO was 3000 ms but measured e2e was ~1 s and predictors stayed under budget → all admitted. To stress admission, use lower `tight-slo-ms` (e.g. 400–800) or longer `max_tokens` / higher concurrency.

## Do **not** fabricate

We do **not** rewrite these numbers to 17%/41%/75% on dual-T4.  
That would be scientific fraud. Framing:

1. **Homogeneous dual-GPU production path** → stable p99 (this run).  
2. **Heterogeneous / slope-skew** → multi-seed sim + optional future mixed GPUs.  
3. Legacy single-seed Locust → secondary only.

## Paper paste lines

```text
On dual Tesla T4 GPUs serving Qwen2.5-3B-Instruct with stock vLLM
(n=3 seeds × 30 requests), DIO-NLMS achieves end-to-end p99 latency of
898±9 ms versus 999±135 ms for Round-Robin and 988±88 ms for Least-Loaded
(100% success). On identical GPUs, routing remains roughly balanced;
NLMS primarily reduces tail variance. In multi-seed slope-skew simulations,
NLMS places 97.5% of requests on the faster worker and reduces p99 by
38.3%±2.0% relative to Round-Robin.
```
