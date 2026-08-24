<p align="center">
  <img src="dio-serve/docs/assets/logo.jpg" alt="DIO" width="140"/>
</p>

<h1 align="center">DIO — a non-invasive control plane for multi-instance LLM serving</h1>

<p align="center">
  <strong>Wrap stock vLLM. Learn latency online. Route smart. Admit safely.</strong><br/>
  Research artifact + installable <code>pip</code> gateway
</p>

<p align="center">
  <a href="dio-serve/README.md"><strong>dio-serve package</strong></a> ·
  <a href="dio-serve/docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="dio-serve/docs/API.md">API docs</a> ·
  <a href="dio-serve/scripts/">Reproduce the paper</a>
</p>

---

**`dio-serve`** is a thin OpenAI-compatible gateway that load-balances across several
**stock vLLM** replicas. No engine patches, no forked vLLM, no custom kernels — it
speaks the HTTP API and reads the Prometheus `/metrics` your engines already expose.

It exists because the usual answer, N vLLM processes behind Nginx or Envoy with
Round-Robin, treats every replica as interchangeable while queue depth, KV-cache
pressure, and transient per-replica slowdowns diverge in practice.

> **Paper:** *DIO: Hybrid Cost Routing, Session Affinity, and Calibration-Robust Admission for
> Multi-Instance LLM Serving over Stock vLLM* — arXiv preprint (link on
> announcement), being revised for a practice-oriented journal submission.

---

## Start here

```bash
git clone https://github.com/nisaral/DIO.git
cd DIO/dio-serve
pip install -e .
dio demo                 # no GPU needed
dio serve -b http://127.0.0.1:8000 -b http://127.0.0.1:8001
```

Point any OpenAI client at `http://localhost:8085/v1`.

Full docs → **[dio-serve/README.md](dio-serve/README.md)** ·
Architecture → **[docs/ARCHITECTURE.md](dio-serve/docs/ARCHITECTURE.md)** ·
API → **[docs/API.md](dio-serve/docs/API.md)**

---

## Architecture

```text
 Clients (OpenAI SDK / curl / LangChain)
                 │
                 ▼
        ┌────────────────┐
        │  DIO Gateway   │  dual-timescale NLMS
        │  :8085 /v1/*   │  joint cost · affinity · admission
        └───────┬────────┘
           HTTP OpenAI API + /metrics scrape (engines unmodified)
        ┌───────┴────────┬────────────┐
        ▼                ▼            ▼
   vLLM GPU0        vLLM GPU1     SGLang / TGI / Ollama
```

DIO **does not** own kernels, KV caches, or continuous batching. Those stay in vLLM.
DIO owns **placement, learning, and admission** — add GPUs by adding URLs.

---

## What it does

**Hybrid cost routing.** Each backend is scored
`S_w = wait_w + ŷ_w + tierCost + vramCost + c_kv·KV_w + c_q·W_w − cacheBonus_w − c_p·Hit_w`.
The latency term `ŷ_w = s_w·N + b_w` comes from a **dual-timescale NLMS** filter
(`s_eff = α·s_fast + (1−α)·s_slow`) that tracks fast shifts without letting noise
destabilize the slow estimate. O(1) per update, no training step.

**Engine-metric fusion.** Scrapes each vLLM's `/metrics` for KV-cache utilization,
waiting-request count, and prefix-hit rate. Read-only.

**Session and prefix affinity.** Multi-turn conversations stay on the replica already
holding their shared prefix, so the KV cache is reused rather than rebuilt. Hit rate
and stickiness are exported, not assumed.

**Admission decoupled from absolute prediction.** Three modes — `rank_only`,
`empirical` (default), `absolute` (diagnostic only; see below).

---

## The main finding is a negative one

A tempting design is to learn one latency model and use it for two jobs: ranking
replicas, and gating admission against an SLO. **The second job does not hold.**

At MAPE ≈ 90–130%, the textbook policy *"reject if min ŷ > SLO"* rejects **~40% of a
load that rank-relative and empirical gating complete in full**. A model accurate
enough to *order* replicas can be far too coarse to *threshold* on.

That is why `absolute` mode ships as a **diagnostic** and the default is `empirical`
(rolling observed percentile). Keeping ranking and admission as separate policies is a
correctness requirement for any predictive gateway built on black-box telemetry, not
an implementation detail.

---

## Measured results

Dual Tesla T4, Qwen2.5-3B-Instruct, two stock vLLM replicas, n=10 seeds.

| Scenario | Result |
|---|---|
| Matched backends | Near Round-Robin parity — no manufactured win |
| Controlled ×2 service-time asymmetry | **p99 −48.3% ± 0.7%** vs RR; beats d=2 RLS |
| Real multi-turn | p99 improves on **8/10 seeds** (median 51.8%, Wilcoxon p ≈ 0.01) |
| Session stickiness | 1.00 vs RR 0.50; **0.998 vs 0.75** under concurrent load |
| Admission, tight SLO | `empirical`/`rank_only` complete all; `absolute` rejects ~40% |

**A caveat we state up front.** In the multi-turn suite the two NLMS arms differ in
*two* knobs at once (engine-metric fusion **and** the affinity cache bonus), so that
margin is a **joint** effect. A decomposition over 100 live snapshots puts the
affinity bonus **194× above** the scraped gauge terms on this hardware, so the gain
belongs to affinity, not the gauges — the reverse of how earlier drafts of our own
work read the same data. Reproduce:
[`scripts/audit_g1_confound.py`](dio-serve/scripts/audit_g1_confound.py).

**Scope.** Two T4s and a 3B model at low concurrency, not an A100/H100 fleet. The
scraped gauge terms contribute little here precisely because queues barely diverge at
that operating point; expect them to matter with deeper concurrency, more replicas, or
heterogeneous peers.

---

## Reproducing the paper

All in [`dio-serve/scripts/`](dio-serve/scripts/):

| Script | Produces |
|---|---|
| `run_gpu_abc_suite.py` | G1–G3 dual-T4 suites (hybrid, affinity, admission) |
| `audit_g1_confound.py` | The 194× decomposition behind the caveat above |
| `run_rls_headtohead.py` | NLMS vs d=2 recursive least squares |
| `run_real_hetero_multiseed.py` | Controlled service-time asymmetry (Regime C) |
| `run_g1_factorial_mock.py` | 2×2 factorial on mock engines, no GPU required |
| `run_abc_experiments.py` | Coefficient and ablation checks |

Runbooks: [`scripts/GPU_ABC_RUNBOOK.md`](dio-serve/scripts/GPU_ABC_RUNBOOK.md) and
[`scripts/GPU_CLUSTER_RUNBOOK.md`](dio-serve/scripts/GPU_CLUSTER_RUNBOOK.md).
Raw multi-seed outputs are committed under `dio-serve/results_*/`.

---

## Key configuration

Every field is settable by flag or `DIO_`-prefixed env var
([`src/dio/config.py`](dio-serve/src/dio/config.py)); paper defaults shown.

| Setting | Default | Meaning |
|---|---|---|
| `--admission-mode` | `empirical` | `empirical` \| `rank_only` \| `absolute` (diagnostic) |
| `--cache-bonus-ms` | `200` | Session/prefix affinity bonus |
| `--slo-ms` | `5000` | Admission budget |
| `engine_metrics` | `true` | Scrape vLLM `/metrics` |
| `kv_cache_cost_ms` | `800` | `c_kv`, × KV utilization |
| `engine_queue_cost_ms` | `50` | `c_q`, × waiting requests |
| `engine_prefix_hit_bonus_ms` | `150` | `c_p`, × prefix-hit rate |

Live introspection while running: `/debug/predictions`, `/debug/affinity`,
`/debug/admission`, `/debug/engine`, `/debug/workers`.

---

## Repository layout

This repo holds **two separate systems**. The paper is about `dio-serve/` only.

| Path | What it is |
|------|-----------|
| **[`dio-serve/`](dio-serve/)** | **The paper's system.** Python/FastAPI gateway over stock vLLM. Start here. |
| [`DIO/`](DIO/) | Earlier, unrelated prototype: Go control plane, BoltDB, gRPC to a Python data plane. Not used for any result in the paper. |
| [`paper_drafts_latex/`](paper_drafts_latex/) | Manuscript sources (Springer submission). |
| [`figs/`](figs/) | Paper figures. |

---

## Citation

Cite the paper, not the software, once the preprint is announced:

```bibtex
@misc{dio2026,
  title  = {DIO: Hybrid Cost Routing, Session Affinity, and Calibration-Robust Admission
            for Multi-Instance LLM Serving over Stock vLLM},
  author = {Nisar, Keyush and Parikh, Krishil and Maisheri, Krisha and
            Gawade, Aruna and Rathod, Nilesh T. and Florence A, Angelin},
  year   = {2026},
  eprint = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  note   = {Software and experimental artifact release}
}
```

## License

Apache-2.0 — see [`dio-serve/LICENSE`](dio-serve/LICENSE).
