# Real dual-T4 GPU A/B/C n=10

## G1 Hybrid
- hybrid_on p99 mean 2000+/-1031 median 1633
- hybrid_off p99 3378+/-43
- rr p99 3388+/-21
- p99 vs RR mean 40.9% +/- 30.5% (wins 8/10 seeds)
- stick hybrid_on 0.998

## G2 Affinity clean
- stick 1.00 hit 0.912 p99 2335+/-852
- RR stick 0.50 p99 3386

## G2b Affinity under load
- stick 0.998 hit 0.932
- RR stick 0.750

## G3 Admission
- absolute rej 0.395+/-0.210
- empirical/rank_only rej 0.0