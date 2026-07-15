# Workshop final suite — honest analysis

**Status:** ok  
**Artifacts:** `summary.json`, `paper_snippets.md`  
**Hardware:** 2× T4, stock vLLM, Qwen2.5-3B-Instruct, HF tokenizer  

## Headlines for paper

### W4 Regime C throttle (n=10) — primary real-GPU win
| Strategy | p99 (ms) | frac e0 | vs RR |
|----------|----------|---------|-------|
| **NLMS** | **3427 ± 38** | 0.47 ± 0.08 | **48.3% ± 0.7%** |
| RLS | 5087 ± 1505 | 0.37 ± 0.19 | 23% ± 23% |
| RR | 6632 ± 37 | 0.50 | — |

### W3 Regime A homogeneous (n=10) — no invented skew
| Strategy | p99 | vs RR | note |
|----------|-----|-------|------|
| NLMS | 3413 ± 31 | **−2.6%** (slightly worse) | tight std |
| RR | 3326 ± 17 | — | |
| RLS | 3466 ± 11 | −4.2% | frac e0 only 0.12 |
| LL | 3313 ± 430 | ~0 | sticky 100% e0 |

### W2 MAPE tokenizer
- heuristic: **89.7 ± 4.9%**
- tokenizer: **117.5 ± 10.9%** (worse)
- → MAPE is **structural**, not a counting bug.

### W5 long decode (max_tokens=128, n=3)
- NLMS ≈ RR (~13830 ms) — fine for homogeneous.

### W6 coeff ±50%
- All variants ~+48–50% vs baseline together → **run-order/cold-start**, not real sensitivity. Do not claim dual-T4 flatness.

## Framing rules
1. Do **not** claim homogeneous p99 win from this suite.
2. Lead with **48% p99 under real throttle n=10**.
3. Report MAPE high + tokenizer worse.
4. RLS loses on real throttle vs NLMS.
5. Earlier n=3 ~900 ms pilot is secondary (different suite scale).
