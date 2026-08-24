# Committed-result reanalysis

Paired bootstrap (100,000 resamples; fixed seed 20260824). Positive deltas mean the baseline p99 is higher.

| Comparison | n | Mean Δ ms | 95% CI Δ ms | Improvement | Cohen dz | Wins |
|---|---:|---:|---:|---:|---:|---:|
| G1 hybrid-on vs round-robin | 10 | 1387.9 | [762.0, 1969.4] | 40.9% | 1.34 | 8/10 |
| G1 hybrid-on vs hybrid-off | 10 | 1378.2 | [767.8, 1949.2] | 41.0% | 1.36 | 8/10 |
| G2 affinity vs round-robin | 10 | 1050.2 | [520.8, 1526.3] | 31.0% | 1.23 | 7/10 |
| G2 load affinity vs round-robin | 10 | 841.7 | [296.8, 1388.3] | 25.0% | 0.90 | 6/10 |
| Regime D NLMS vs round-robin | 5 | 2537.2 | [1796.7, 3681.8] | 33.3% | 1.95 | 5/5 |
| Regime D NLMS vs least-loaded | 5 | 4263.8 | [3498.2, 5518.2] | 45.5% | 3.10 | 5/5 |
