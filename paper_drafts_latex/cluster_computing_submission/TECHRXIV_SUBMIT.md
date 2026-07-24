# TechRxiv preprint — copy-paste kit for DIO

**Upload this PDF:**  
`paper_drafts_latex/cluster_computing_submission/DIO_ClusterComputing.pdf`

**Portal:** https://www.techrxiv.org/  
**Login / submit:** create free IEEE account if needed, then **Submit a Preprint**

---

## Step-by-step

1. Open https://www.techrxiv.org/ and sign in (IEEE account is free).
2. Click **Submit a Preprint** (or **Upload** / **New submission**).
3. Drag-and-drop **`DIO_ClusterComputing.pdf`**.
4. Paste the fields below exactly.
5. Confirm authors/emails match the PDF.
6. Choose license: prefer **CC BY 4.0** (or whatever TechRxiv offers that allows later journal submission).
7. Submit → wait for **moderation** (screening, not full peer review).  
   When live you get a **DOI** like `10.36227/techrxiv....`

---

## Title

```
DIO: Dual-Timescale Predictive Orchestration for Heterogeneous LLM Inference Clusters
```

---

## Authors (order as in paper)

| Order | Name | Email | Affiliation |
|-------|------|-------|-------------|
| 1 (corresponding) | Keyush Nisar | Nisarkeyush3@gmail.com | Dwarkadas J. Sanghvi College of Engineering, Mumbai, India |
| 2 | Krishil Parikh | Krishilparikh75@gmail.com | Dwarkadas J. Sanghvi College of Engineering, Mumbai, India |
| 3 | Krisha Maisheri | KrishaMaisheri16@gmail.com | Dwarkadas J. Sanghvi College of Engineering, Mumbai, India |

---

## Abstract (paste)

```
Large language model (LLM) serving stacks maximize single-node throughput but underperform when multiple GPUs differ in speed, queueing, or memory headroom. We present DIO, a non-invasive cluster control plane that sits in front of unmodified OpenAI-compatible engines (e.g., stock vLLM). DIO learns per-worker ms/token slopes online with dual-timescale Normalized Least Mean Squares (NLMS) using O(1) arithmetic per completion, ranks workers by a joint cost (service estimate, queue wait, tier, VRAM pressure, cache affinity), and applies Roofline-inspired hard blocks without trusting absolute latency predictions for admission. The system is released as dio-serve, a pip-installable gateway and library.

Under fixed hardware, DIO's contribution is relative cost ranking for routing, not point-accurate millisecond prediction. On dual Tesla T4 GPUs with Qwen2.5-3B-Instruct and stock vLLM, multi-seed evaluation (n=10) shows that when one peer is delay-throttled (x2 observed e2e), NLMS reduces p99 end-to-end latency by 48.3%+/-0.7% versus Round-Robin and outperforms a d=2 RLS predictor; when both GPUs are identical, strategies are nearly tied, as expected. Absolute MAPE remains high (approximately 90-130%). We release code, validation suites, and dual-T4 artifacts for reproduction.
```

---

## Keywords / tags (pick all that apply)

```
LLM serving
cluster scheduling
heterogeneous GPUs
load balancing
NLMS
vLLM
distributed systems
inference systems
open source
```

---

## Category / subject (choose closest)

Prefer something like:

- **Computer Science and Engineering**
- **Computing and Processing** / **Distributed Computing**
- **Artificial Intelligence** (secondary if allowed)

If the form has free text, use:  
`Computer Science — Distributed Systems / LLM Inference Serving`

---

## Optional notes / comments

```
Systems technical report. Open-source implementation: https://github.com/nisaral/DIO (dio-serve wraps stock vLLM and other OpenAI-compatible engines). Multi-seed dual-T4 evaluation included. Preprint; journal submission may follow.
```

---

## After it goes live

1. Copy the **TechRxiv DOI / URL**.
2. Add to GitHub README, e.g.:

```markdown
## Paper
Preprint: [TechRxiv DOI](https://doi.org/10.xxxx/...)  
Code: this repository (`dio-serve`)
```

3. You can still submit the **same work** later to Cluster Computing (or another journal); **disclose the TechRxiv link** in the cover letter.

---

## Checklist before click Submit

- [ ] PDF is `DIO_ClusterComputing.pdf` (latest build)
- [ ] All 3 authors listed with emails
- [ ] Abstract pasted without weird line breaks
- [ ] Keywords filled
- [ ] GitHub link in notes
- [ ] You understand: preprint ≠ peer-reviewed journal, but still citable + later journal OK
