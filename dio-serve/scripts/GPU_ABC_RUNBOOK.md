# Real-GPU A/B/C suite (title contributions)

Validates **hybrid cost**, **session affinity**, and **admission** on dual stock
vLLM — not only the library multi-seed harness.

## What it runs

| Phase | What | Real GPU? |
|-------|------|-----------|
| G1 Hybrid | NLMS+metrics ON vs OFF vs RR, multi-turn long system prefix + light concurrency | Yes |
| G2 Affinity | NLMS+affinity bonus vs RR, multi-turn stickiness + DIO affinity hit rate | Yes |
| G3 Admission | absolute vs empirical vs rank_only with tight SLO on real gateway | Yes |

## Kaggle dual-T4

```bash
cd /kaggle/working/DIO   # or your clone path
git pull origin feat-v2-rls-scheduler
cd dio-serve
pip install -e . -q

python scripts/run_gpu_abc_suite.py \
  --engine-mode vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --gpus 0,1 \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --seeds 5 \
  --sessions 12 \
  --turns 4 \
  --max-tokens 32 \
  --concurrent 2 \
  --adm-slo-ms 800 \
  --out /kaggle/working/results_gpu_abc
```

Quick smoke (2 seeds):

```bash
python scripts/run_gpu_abc_suite.py --quick --engine-mode vllm --gpus 0,1 \
  --model Qwen/Qwen2.5-3B-Instruct --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --out /kaggle/working/results_gpu_abc_quick
```

## External engines already running

```bash
python scripts/run_gpu_abc_suite.py \
  --backends http://127.0.0.1:8000,http://127.0.0.1:8001 \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --seeds 5 \
  --out results_gpu_abc
```

## Outputs

- `results_gpu_abc/summary.json` — full multi-seed rows
- `results_gpu_abc/paper_snippets.md` — paste-ready bullets
- `results_gpu_abc/logs/` — vLLM + DIO logs

## Paper paste targets

| Suite | Paper home |
|-------|------------|
| G1 hybrid p99 / stickiness | §7.7 Table hybrid + figure |
| G2 stickiness / affinity hit | §7.8 Table affinity |
| G3 reject rates / abs disagree | §7.9 Table admission |

Label clearly: **real dual-T4 multi-turn** vs library suite.

## Journal strengthen pass (reviewer asks)

```bash
# n=10 primary + affinity under load (G2b). Same harness, more seeds.
python scripts/run_gpu_abc_suite.py \
  --engine-mode vllm --gpus 0,1 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --seeds 10 \
  --affinity-concurrent 2 \
  --out /kaggle/working/results_gpu_abc_n10

# Optional dual-T4 hybrid coeff ±50% (expensive)
python scripts/run_gpu_abc_suite.py --backends ... --seeds 5 --run-coeff --skip-g2 --skip-g2b --skip-g3 \
  --out results_gpu_abc_coeff
```

## Priority if GPU time is short

1. **Bump seeds to 10** on G1–G3 (same scripts; biggest confidence gain)
2. **G2b affinity under concurrent load** (now default unless `--skip-g2b`)
3. **G3 admission** (tight `--adm-slo-ms`)
4. **G4 `--run-coeff`** only if time remains

## Notes

- Shadow MAPE under RR still appears in debug metrics (predictor always updates).
- If `/metrics` probe fails, hybrid ON ≈ hybrid OFF; check vLLM Prometheus flags.
- Admission SLO must be in the same order of magnitude as real e2e (e.g. 800–3000 ms on T4 3B short decode).
