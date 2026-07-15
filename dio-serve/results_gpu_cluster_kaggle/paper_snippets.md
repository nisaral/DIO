# Paper-ready snippets (from GPU cluster validation)
Generated: 2026-07-15T14:22:22.434400+00:00
Model: Qwen/Qwen2.5-3B-Instruct
Engine mode: vllm

## G2 multi-seed e2e p99 (mean±std)
- **nlms**: $898.4 \pm 8.8$ ms (n=3 seeds)
- **round_robin**: $998.7 \pm 134.9$ ms (n=3 seeds)
- **least_loaded**: $988.5 \pm 88.3$ ms (n=3 seeds)

## G3 heterogeneity
- NLMS fraction to first (fast) backend: $0.433 \pm 0.000$
- p99 improvement vs RR: $7.8\% \pm 1.6\%$

## G4 dual vs single
- dual_p99: $1067.56 \pm 33.51$
- single_p99: $1071.22 \pm 16.50$
- dual_mape: $81.63 \pm 0.94$
- single_mape: $85.94 \pm 0.48$
