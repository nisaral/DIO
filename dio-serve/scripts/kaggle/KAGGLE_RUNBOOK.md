# Kaggle dual-T4 follow-ups (EWMA + concurrency)

**Purpose:** close Reviewer gaps that do **not** need T4+A30 physical SKU:

| Gap | What to run | Paper use |
|-----|-------------|-----------|
| **EWMA baseline** | dual T4, matched + optional ×2 delay-proxy | “NLMS vs simple recent-latency ranker” |
| **Concurrency** | dual T4, concurrent 1/4/8 | “ranking survives multi-in-flight” |
| **More seeds** | optional `--seeds 10` on same cells | stronger p99 / win counts |

**Do not put mock-engine tables in the paper.** Mock is laptop-only mechanism smoke.

Kaggle GPU: pick **2× T4** (or 2× P100 if that is all you get — note SKU in summary).

---

## 0. One-time notebook setup

```python
# Cell 1 — clone + install (adjust branch/repo if needed)
!nvidia-smi
!git clone --depth 1 -b feat-v2-rls-scheduler https://github.com/nisaral/DIO.git /kaggle/working/DIO
%cd /kaggle/working/DIO/dio-serve
!pip install -e . -q
!pip install "vllm==0.8.5" "transformers==4.51.3" -q
```

```python
# Cell 2 — HF cache (optional, speeds reload)
import os
os.environ["HF_HOME"] = "/kaggle/working/hf"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf"
```

---

## 1. EWMA baseline (PRIORITY — validates NLMS structure)

Compares **`nlms` vs `ewma` vs `round_robin` vs `rls`** on the **same dual engines**:

- **Matched** dual-T4 (homogeneous): expect near-RR parity for all learners  
- **Skew** (delay-proxy ×2 on e1): expect NLMS/EWMA/RLS to beat RR; NLMS should not collapse vs EWMA  

```bash
cd /kaggle/working/DIO/dio-serve

python scripts/kaggle/run_kaggle_followups.py \
  --mode ewma \
  --engine-mode vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --gpus 0,1 \
  --seeds 10 \
  --n-per-seed 30 \
  --max-tokens 32 \
  --slow-mult 2.0 \
  --out /kaggle/working/results_kaggle_ewma
```

**Quick smoke (1–2 seeds):**

```bash
python scripts/kaggle/run_kaggle_followups.py \
  --mode ewma --quick --gpus 0,1 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --out /kaggle/working/results_kaggle_ewma_smoke
```

**Artifacts:**

- `/kaggle/working/results_kaggle_ewma/summary.json`  
- `paper_snippets.md` (LaTeX-ready means ± std)

**How to read it for the paper:**

- If **EWMA ≈ NLMS** on p99 under skew → claim *lightweight black-box ranking* (structure optional), not “NLMS unique.”  
- If **NLMS > EWMA** under mixed `max_tokens` or skew → claim *token-aware dual-timescale model helps.*  
- Homogeneous: all should be near RR (same as Regime A story).

---

## 2. Concurrency sweep (PRIORITY #2 — if session time remains)

```bash
cd /kaggle/working/DIO/dio-serve

python scripts/kaggle/run_kaggle_followups.py \
  --mode concurrency \
  --engine-mode vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --gpus 0,1 \
  --seeds 5 \
  --n-per-seed 40 \
  --concurrencies 1,4,8 \
  --strategies nlms,round_robin,least_loaded,ewma \
  --out /kaggle/working/results_kaggle_concurrency
```

Reports per concurrency: p50/p99, frac e0, throughput, vs-RR p99.

**Honest paper wording:** dual-T4 concurrency mechanism, **not** Regime D multi-SKU concurrent load.

---

## 3. Optional: more seeds on existing workshop matrix

If you only want extra n without EWMA:

```bash
python scripts/run_workshop_final_suite.py \
  --engine-mode vllm \
  --model Qwen/Qwen2.5-3B-Instruct \
  --tokenizer Qwen/Qwen2.5-3B-Instruct \
  --gpus 0,1 \
  --strategies-a nlms,rls,ewma,round_robin,least_loaded \
  --strategies-c nlms,rls,ewma,round_robin \
  --out /kaggle/working/results_workshop_with_ewma
```

(`ewma` is a first-class `--strategy` after the dual-timescale NLMS patch.)

---

## 4. Download results to laptop

In notebook:

```python
!cd /kaggle/working && tar czf kaggle_dio_followups.tgz results_kaggle_ewma results_kaggle_concurrency 2>/dev/null; ls -lh kaggle_dio_followups.tgz
```

Then **Notebook → Data → Download** the tarball, or copy via Kaggle API.

On laptop, drop under `dio-serve/results_kaggle_ewma/` and we can paste numbers into the paper.

---

## 5. Session tips

| Tip | Why |
|-----|-----|
| Pin `vllm==0.8.5` + `transformers==4.51.3` | Matches Regime D / dual-T4 paper pin |
| Use `--dtype half` if script exposes it (default in helper) | T4 has no bf16 |
| Don’t enable prefix-caching unless measuring affinity | Keeps EWMA cell clean |
| One mode per session if flaky | Restart notebook, re-run only `--mode ewma` |
| Kaggle wall time | EWMA matched+skew n=10 ≈ 1–2 h; concurrency n=5 ≈ 1 h |

---

## 6. What still needs *non-Kaggle* cloud

- True **T4+A30** concurrent Regime D  
- Production Stack **GPU** bake-off  

Kaggle dual-T4 **does** answer: EWMA necessity + concurrent ranking on real GPUs.
