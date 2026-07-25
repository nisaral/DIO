# arXiv submit kit — DIO systems paper

**PDF to upload:**  
`paper_drafts_latex/cluster_computing_submission/DIO_ClusterComputing.pdf`

**Portal:** https://arxiv.org/login → Start new submission

---

## Categories

| Role | Category |
|------|----------|
| **Primary** | **cs.DC** (Distributed, Parallel, and Cluster Computing) |
| Cross-list | **cs.LG** (Machine Learning) |
| Optional | cs.PF (Performance) |

If first time in cs.DC, arXiv may ask for **endorsement** — follow their email, or use a coauthor who is already endorsed.

---

## Title

```
DIO: A Non-Invasive Control Plane for Multi-Instance LLM Serving over Stock vLLM
```

---

## Authors

```
Keyush Nisar
Krishil Parikh
Krisha Maisheri
```

Affiliations: Dwarkadas J. Sanghvi College of Engineering, Mumbai, India  
Emails: Nisarkeyush3@gmail.com, Krishilparikh75@gmail.com, KrishaMaisheri16@gmail.com

---

## Abstract (paste into form)

```
Deployments often place several stock vLLM (or OpenAI-compatible) instances behind Round-Robin or connection-count balancers. That works poorly when backends differ in effective service cost—queue depth, interference, or a temporarily slow peer—even when GPU SKUs match. We present DIO/dio-serve, a non-invasive OpenAI-compatible gateway that ranks backends with dual-timescale NLMS slopes learned from request end-to-end latency and token counts (plus optional VRAM hints and in-gateway queue depth), not full GPU SM/power telemetry.

Honest scope: absolute latency prediction is weak (MAPE ~90-130% on dual-T4); DIO is useful as a relative ranker under measurable service skew. On dual Tesla T4 + Qwen2.5-3B + stock vLLM (n=10 seeds): (i) when backends are identical, NLMS p99 is not better than Round-Robin (within ~3%, slightly worse); (ii) when one peer's observed e2e is delay-inflated x2, NLMS cuts p99 by 48.3%+/-0.7% vs RR and beats d=2 RLS. We release open code and multi-seed artifacts. Multi-SKU fleets are out of scope.
```

---

## Comments (optional)

```
14 pages. Code: https://github.com/nisaral/DIO (dio-serve wraps stock vLLM). Systems technical report on multi-instance LLM routing; honest evaluation when backends match vs under service-time skew.
```

---

## License

Prefer **CC BY 4.0** (or arXiv non-exclusive distribution license).  
Later journal (e.g. Cluster Computing) is still OK if you disclose the arXiv ID.

---

## Steps

1. https://arxiv.org/login  
2. **Start new submission** → Computer Science  
3. Primary **cs.DC**, add **cs.LG**  
4. Upload **DIO_ClusterComputing.pdf**  
5. Paste title, authors, abstract, comments  
6. Preview → Submit  
7. Wait for announcement (often next business day US Eastern)

---

## After you get arXiv ID (e.g. 2607.xxxxx)

Add to GitHub README:

```markdown
## Paper
Preprint: https://arxiv.org/abs/XXXX.XXXXX  
Code: this repo (`dio-serve`)
```

You can still submit to **Cluster Computing** later; put the arXiv link in the cover letter.

---

## Checklist

- [ ] Latest PDF (honest multi-instance reframe)
- [ ] cs.DC + cs.LG
- [ ] All authors agree
- [ ] Not under review at two peer-reviewed venues at once (arXiv is fine)
- [ ] Code link in comments
