# Post DIO as an arXiv systems paper (free, ~15 minutes)

You already have a real systems artifact (vLLM wrap, multi-seed dual-T4, open `dio-serve`).  
**arXiv is the right “ship it” move** — no APC, citable, standard for systems work.

## 1. Categories (primary + secondary)

| Slot | Category | Why |
|------|----------|-----|
| **Primary** | **cs.DC** (Distributed, Parallel, and Cluster Computing) | Cluster orchestration / multi-worker serving |
| Cross-list | **cs.LG** (Machine Learning) | LLM inference systems |
| Optional | **cs.OS** or **cs.PF** | Scheduling / performance |

## 2. Files to upload

From `paper_drafts_latex/cluster_computing_submission/`:

1. **`DIO_ClusterComputing.pdf`** (main)
2. Optional: source zip (`.tex` + `.bib` + figures) if you want reproducibility of the paper build

Also link code: `https://github.com/nisaral/DIO`

## 3. Title (use as-is or shorten)

```
DIO: Dual-Timescale Predictive Orchestration for Heterogeneous LLM Inference Clusters
```

## 4. Abstract (paste into arXiv form)

```
Large language model (LLM) serving stacks maximize single-node throughput but underperform when multiple GPUs differ in speed, queueing, or memory headroom. We present DIO, a non-invasive cluster control plane that sits in front of unmodified OpenAI-compatible engines (e.g., stock vLLM). DIO learns per-worker ms/token slopes online with dual-timescale Normalized Least Mean Squares (NLMS) using O(1) arithmetic per completion, ranks workers by a joint cost (service estimate, queue wait, tier, VRAM pressure, cache affinity), and applies Roofline-inspired hard blocks without trusting absolute latency predictions for admission. The system is released as dio-serve, a pip-installable gateway and library.

Under fixed hardware, DIO's contribution is relative cost ranking for routing, not point-accurate millisecond prediction. On dual Tesla T4 GPUs with Qwen2.5-3B-Instruct and stock vLLM, multi-seed evaluation (n=10) shows that when one peer is delay-throttled (x2 observed e2e), NLMS reduces p99 end-to-end latency by 48.3%+/-0.7% versus Round-Robin and outperforms a d=2 RLS predictor; when both GPUs are identical, strategies are nearly tied, as expected. Absolute MAPE remains high (~90-130%). We release code, validation suites, and dual-T4 artifacts for reproduction.
```

## 5. Comments field (optional)

```
14 pages. Code: https://github.com/nisaral/DIO (dio-serve). Systems technical report: predictive routing for multi-GPU LLM serving over stock vLLM.
```

## 6. Submit steps

1. Create/login: https://arxiv.org/login  
2. **Start new submission** → Computer Science  
3. Primary: **cs.DC**, cross-list **cs.LG**  
4. Upload PDF  
5. Paste title, authors, abstract  
6. License: recommend **CC BY 4.0** or arXiv’s non-exclusive distribution  
7. Submit → wait for announce (usually next business day)

**First-time in cs.DC?** You may need an **endorsement** (arXiv emails instructions). Ask a colleague with arXiv history, or submit via an endorsed coauthor account.

## 7. After it posts

- Put the arXiv link in `README.md`  
- Tag a GitHub release: `v0.2-paper`  
- Optional later: still submit to Cluster Computing using the same PDF (cite arXiv)

## 8. What this is *not*

- Not a peer-reviewed journal accept (yet)  
- Still a **real systems paper**: integrated with vLLM, multi-seed GPU results, open software  

That’s enough to claim the work, get citations, and park the idea while you move on.
