# arXiv submit kit — DIO

**PDF:** `DIO_ClusterComputing.pdf` (21 pages, builds clean, 0 undefined refs)
**Portal:** https://arxiv.org/login → Start new submission → Computer Science

---

## Categories

| Role | Category |
|------|----------|
| **Primary** | **cs.DC** — Distributed, Parallel, and Cluster Computing |
| Cross-list | **cs.PF** — Performance |
| Optional | cs.LG — Machine Learning |

cs.PF is the better cross-list than cs.LG: the contribution is measurement and
scheduling, not a learning method. If this is a first cs.DC submission arXiv may
ask for **endorsement** — a coauthor already endorsed in cs.DC can submit instead.

---

## Title

```
DIO: Hybrid Cost Routing, Session Affinity, and Sound Admission for
Multi-Instance LLM Serving over Stock vLLM
```

---

## Authors (in paper order — all six, all equal main authors)

```
Keyush Nisar, Krishil Parikh, Krisha Maisheri, Aruna Gawade, Nilesh T. Rathod, Angelin Florence A
```

Affiliation (all): Department of Artificial Intelligence and Machine Learning, Dwarkadas J. Sanghvi
College of Engineering, Mumbai, India

Submitting/contact author: Keyush Nisar — nisarkeyush3@gmail.com

---

## Abstract — primary version (paste this one)

Leads with the negative result, because that is the transferable finding and it is
what makes the paper worth reading rather than a speedup report.

```
Operators commonly run several stock vLLM instances behind Round-Robin or
least-connections balancers. Those balancers treat every peer as equal, yet queue
depth, KV-cache pressure, and transient slowdowns diverge across replicas. A
tempting fix is to learn a per-worker latency model and use it both to rank
replicas and to gate admission against an SLO. We show the second half of that
plan does not hold.

We present DIO (dio-serve), a thin OpenAI-compatible gateway requiring no engine
patches. DIO ranks backends with dual-timescale NLMS slopes learned from
end-to-end latency and token counts, adjusts scores with the Prometheus /metrics
each vLLM instance already exports, keeps multi-turn sessions on the replica
holding the shared prefix, and deliberately decouples admission from absolute
predicted latency.

On dual Tesla T4 GPUs with Qwen2.5-3B and stock vLLM (n=10 seeds): when backends
are matched, NLMS ranking stays near Round-Robin parity rather than inventing
wins; under controlled 2x service-time asymmetry it cuts p99 by 48.3% +/- 0.7%
against Round-Robin and beats d=2 recursive least squares; on real multi-turn
traffic, enabling session affinity together with engine-metric fusion improves
p99 on 8/10 seeds (median 51.8%, Wilcoxon p ~ 0.01), and a decomposition
attributes that gain to the affinity cache bonus rather than the scraped gauges,
which are 194x smaller on this hardware.

Our main finding is negative and, we argue, more transferable than any of the
above. The same online model that ranks replicas well is far too poorly
calibrated in absolute terms to threshold on: at MAPE ~90-130%, the textbook
"reject if min predicted latency > SLO" policy rejects ~40% of a load that
rank-relative and empirical-percentile gating complete in full. Keeping ranking
and admission as separate policies is therefore a correctness requirement, not an
implementation detail, for any predictive gateway built on black-box telemetry.
Code and multi-seed artifacts: https://github.com/nisaral/DIO
```

---

## Abstract — shorter alternate (if you want it tighter)

```
Operators commonly place several stock vLLM instances behind Round-Robin, which
treats every peer as equal even though queues, KV-cache pressure, and transient
slowdowns diverge. We present DIO (dio-serve), a thin OpenAI-compatible gateway
that needs no engine patches: it ranks backends with dual-timescale NLMS, fuses
the Prometheus /metrics vLLM already exports, pins multi-turn sessions to the
replica holding the shared prefix, and decouples admission from absolute
predicted latency.

On dual Tesla T4 GPUs with Qwen2.5-3B and stock vLLM (n=10 seeds), DIO holds
Round-Robin parity when backends match, cuts p99 by 48.3% +/- 0.7% under
controlled 2x service-time asymmetry, and improves multi-turn p99 on 8/10 seeds
(median 51.8%, Wilcoxon p ~ 0.01) -- a gain a decomposition attributes to session
affinity rather than the scraped gauges.

We also report a negative result we consider the most transferable finding here: a
latency model good enough to rank replicas can be far too coarse to threshold on.
At MAPE ~90-130% the natural "reject if min predicted latency > SLO" gate rejects
~40% of a load that rank-relative and empirical gating serve in full. Ranking and
admission must stay separate policies. Code: https://github.com/nisaral/DIO
```

---

## Comments field

```
21 pages, 7 figures, 8 tables. Under review at Cluster Computing (Springer).
Code and multi-seed artifacts: https://github.com/nisaral/DIO
```

State "under review", never "accepted" or "published". Update the arXiv entry
with the journal reference once (and only once) it is actually accepted.

---

## License

**CC BY 4.0** is the safe choice and stays compatible with Springer's preprint
policy. If you would rather keep the tightest option, arXiv's non-exclusive
distribution licence also works. Do not pick a no-derivatives licence — some
journals object.

---

## Order of operations

Post to arXiv **first**, then submit to Cluster Computing with the arXiv ID in the
cover letter. Springer permits preprints, and having the ID in hand means the
cover letter can cite it. Announcement is usually the next US-Eastern business day.

1. https://arxiv.org/login → Start new submission → Computer Science
2. Primary cs.DC, cross-list cs.PF
3. Upload `DIO_ClusterComputing.pdf`
4. Paste title / all six authors / abstract / comments
5. Pick CC BY 4.0
6. Preview, check the PDF renders and the author list is complete, Submit

---

## After the arXiv ID lands

Add to the GitHub README, near the top:

```markdown
## Paper
Preprint: https://arxiv.org/abs/XXXX.XXXXX (under review, Cluster Computing)
Reproduce: see `scripts/` for the G1-G3 dual-T4 suites
```

---

## Pre-submission checklist

- [ ] All six coauthors have agreed to the arXiv posting (required — not optional)
- [ ] GitHub repo description matches the paper. **Currently it does not**: it
      describes a Go/BoltDB/gRPC MLOps orchestrator, while the paper describes a
      FastAPI gateway over stock vLLM. Fix before posting; reviewers click.
- [ ] `dio-serve/` and the G1-G3 reproduction scripts are obvious from the repo
      landing page
- [ ] PDF is the current 21-page build
- [ ] Not under review at two peer-reviewed venues simultaneously (arXiv does not
      count as one)

