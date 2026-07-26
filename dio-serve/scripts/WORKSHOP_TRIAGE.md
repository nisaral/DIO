# Workshop triage — remaining Kaggle / cloud cells

## ONE SCRIPT (preferred)

All outstanding dual-T4 re-runs in one job (no mixed SKU needed):

```bash
cd dio-serve
pip install -e . -q
python scripts/run_workshop_final_suite.py \
  --engine-mode vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --gpus 0,1 \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --out results_workshop_final
```

Kaggle notebook cell:

```python
!cd /kaggle/working/DIO/dio-serve && git pull origin feat-v2-rls-scheduler && pip install -e . -q

!python scripts/run_workshop_final_suite.py \
  --engine-mode vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --gpus 0,1 \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --out /kaggle/working/results_workshop_final
```

### What it runs

| Phase | What | Default |
|-------|------|---------|
| W2 | MAPE heuristic vs HF tokenizer | 3 seeds |
| W3 | Regime A n=10 NLMS/RLS/RR/LL | 10×30, max_tokens=32 |
| W4 | Regime C delay-proxy ×2 n=10 NLMS/RLS/RR | 10×30 |
| W5 | Longer decode multi-seed | n=3, max_tokens=128 |
| W6 | Coefficient ±50% live sweep | 7 variants |

### Outputs

```
results_workshop_final/
  summary.json
  paper_snippets.md
  logs/
```

### Options

```bash
# Short smoke (2 seeds, fewer reqs)
python scripts/run_workshop_final_suite.py --quick --gpus 0,1 ...

# Only some phases
python scripts/run_workshop_final_suite.py --only w2,w3,w4 --gpus 0,1 ...

# Engines already up
python scripts/run_workshop_final_suite.py --engine-mode external \
  --backends http://127.0.0.1:18000,http://127.0.0.1:18001 ...
```

**Runtime note:** full suite is long (many seeds × strategies × vLLM). Prefer a dual-T4 session with several hours. Use `--quick` first to verify, then full run.

## Optional paid

1 hour L4/A10 on RunPod/Vast + Kaggle T4 for true mixed-SKU (highest credibility/$). Not required for W2–W6.
