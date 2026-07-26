# Real dual-T4 hetero (delay proxy) — multi-seed

**Status:** ok  
**Script:** `run_real_hetero_multiseed.py`  
**Hardware:** 2× Tesla T4, stock vLLM, Qwen2.5-3B-Instruct  
**Skew:** e1 behind `latency_delay_proxy` ×2.0 (real tokens, inflated e2e)  
**Seeds:** 5 × 30 requests, max_tokens=32  

## Headline (paper Table real_hetero)

| Metric | NLMS | RR |
|--------|------|-----|
| Frac to fast (e0) | **0.680 ± 0.073** | 0.500 ± 0.000 |
| e2e p99 (ms) | **2058 ± 237** | 3196 ± 747 |
| p99 improvement | **34.3% ± 7.1%** | — |

- Directional ranking on **real** engines: NLMS prefers fast GPU (~68%) vs RR 50%.  
- Large tail win despite high MAPE (~80–95% on short seeds).  
- Weaker skew than CPU sim (97.5%) — expected with cold start + only 30 req/seed.

## Honest notes

- Delay proxy multiplies **observed** wall time, not SM clocks.  
- Not a mixed-SKU fleet (T4+L4); validates ranking under unequal service times.  
- Do not claim 97.5% on dual-T4; report 68% ± 7.3% for real throttle.

## Artifacts

- `summary.json`, `paper_snippets.md`
