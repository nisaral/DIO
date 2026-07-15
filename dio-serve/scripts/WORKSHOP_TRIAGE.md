# Workshop triage — remaining Kaggle / cloud cells

Code already shipped for #1 admission modes, #2 RLS harness, #6 tokenizer.
Run these on Kaggle after `git pull` + `pip install -e .`.

## A) RLS head-to-head on real dual-T4 (Priority 1)

Homogeneous (Regime A style):

```bash
python scripts/run_rls_headtohead.py --mode real \
  --gpus 0,1 --model Qwen/Qwen2.5-3B-Instruct \
  --seeds 3 --requests-per-seed 30 --max-tokens 32 \
  --slow-mult 1.0 \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --strategies nlms,rls,round_robin,least_loaded \
  --out results_rls_headtohead_real
```

Throttled hetero (Regime C style):

```bash
python scripts/run_rls_headtohead.py --mode real \
  --gpus 0,1 --model Qwen/Qwen2.5-3B-Instruct \
  --seeds 5 --requests-per-seed 30 --max-tokens 32 \
  --slow-mult 2.0 \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --strategies nlms,rls,round_robin,least_loaded \
  --out results_rls_headtohead_real_hetero
```

## B) More seeds (Priority 4)

```bash
# Regime A n=10
python scripts/run_gpu_cluster_validation.py \
  --engine-mode vllm --model Qwen/Qwen2.5-3B-Instruct --gpus 0,1 \
  --seeds 10 --requests-per-seed 30 --max-tokens 32 \
  --strategies nlms,rls,round_robin,least_loaded \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --skip-g3 --skip-g4 --skip-g5 \
  --out results_regime_a_n10

# Regime C real throttle n=10
python scripts/run_real_hetero_multiseed.py \
  --gpus 0,1 --seeds 10 --requests-per-seed 30 --slow-mult 2.0 \
  --out results_gpu_cluster_hetero_n10
```

## C) Longer decode (Priority 6)

```bash
python scripts/run_gpu_cluster_validation.py \
  --engine-mode vllm --model Qwen/Qwen2.5-3B-Instruct --gpus 0,1 \
  --seeds 3 --requests-per-seed 20 --max-tokens 128 \
  --strategies nlms,round_robin \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --skip-g3 --skip-g4 --skip-g5 \
  --out results_long_decode_mt128
```

## D) Tokenizer MAPE re-measure (Priority 2 — free if bundled with A)

Use `--tokenizer Qwen/Qwen2.5-3B-Instruct` on any of the above; compare MAPE in `summary.json` vs heuristic baseline.

## Already done in code/paper (no GPU required)

| Item | Status |
|------|--------|
| #1 Admission empirical/rank_only + absolute diagnostic | shipped; paper §admission_decouple |
| #2 RLS sim head-to-head table | `results_rls_headtohead/` + Table rls_h2h |
| #6 HF tokenizer path | `--tokenizer` / `DIO_TOKENIZER_NAME` |
| #4 Cross-SKU reframe | top limitation in paper |
| #3 Power caveat | threats § |

## Optional paid

1 hour L4/A10 on RunPod/Vast + Kaggle T4 for true mixed-SKU (highest credibility/$).
