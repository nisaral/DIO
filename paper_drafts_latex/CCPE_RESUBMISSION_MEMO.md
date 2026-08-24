# DIO resubmission assessment for CCPE

## Bottom line

The paper is publishable as a focused systems/practice paper, but not with the
headline that it is a generally superior LLM scheduler. The strongest defensible
story is narrower and more interesting:

> A deployable, engine-agnostic control loop can learn relative backend capacity
> from black-box completions, preserve prefix locality, and make admission safe
> even when its absolute latency model is badly calibrated.

The repository already contains a missing piece of evidence: a five-seed real
T4/A30 experiment. NLMS reduces p99 by 33.3% versus Round-Robin on all five seeds;
Least-Loaded sends all traffic to the T4 and is 22.7% worse than Round-Robin. This
is materially stronger than relying only on an injected 2x delay between matched
T4s, and it has now been added to the manuscript source.

## Likely reasons for desk rejection

1. The mechanisms individually look incremental: NLMS, metric-weighted routing,
   affinity, and admission are known ideas. Novelty must be claimed for the
   deployable composition and the measured separation of ranking from admission,
   not for each ingredient.
2. The earlier heterogeneity evidence was mostly a delay proxy on identical GPUs.
   Editors could reasonably read this as insufficient maturity for a cluster journal.
3. Prediction MAPE is 90--130%. Without the negative-calibration framing this looks
   like a broken predictor, even though rank ordering is useful.
4. The G1 `hybrid on/off` comparison is confounded: the affinity/cache term dominates
   the scraped gauge terms on this hardware. The paper must not call the full gain a
   Prometheus-metric gain.
5. The evaluation is two replicas, mostly a 3B model, low concurrency, and no direct
   executable comparison with vLLM Production Stack or another prefix-aware router.
6. The manuscript is broad and long. The legacy single-seed Locust section weakens
   the otherwise multi-seed empirical standard and should move to an appendix or be
   removed for CCPE.

## Recommended novelty framing

Use three contributions, not a list of loosely novel features:

1. **Observability-aware black-box routing.** One HTTP gateway uses completed-request
   feedback plus optional stock engine telemetry to infer effective service capacity
   without engine patches. Its value is shown under both transient slowdown and a
   real T4/A30 deployment.
2. **Rank/calibration separation.** Demonstrate that a high-MAPE model can still be a
   useful ranker but is unsafe as an admission threshold. The empirical admission
   policy is the architectural consequence of that finding.
3. **Locality-preserving deployment path.** Prefix/session affinity is integrated at
   the gateway, with explicit stickiness and affinity-hit observability, while the
   underlying engines remain stock and independently upgradeable.

Avoid claiming that scraping `/metrics`, prefix affinity, NLMS itself, or a weighted
sum is independently novel.

## Highest-value improvements requiring no new GPU time

1. Promote the existing T4/A30 Regime D to the main evaluation and artifact index.
2. Reanalyse all committed per-request results with paired bootstrap confidence
   intervals and effect sizes; do not depend only on mean +/- standard deviation.
3. Turn the existing Production Stack smoke artifacts into a clearly labelled
   compatibility/deployment case study. Do not present mock-engine numbers as a
   performance comparison.
4. Add a threat-to-validity table mapping every claim to hardware, seeds, workload,
   baseline, and artifact path.
5. Remove or appendix the legacy single-seed Locust pilot.
6. Rename “sound admission” to “calibration-robust admission” unless a formal safety
   theorem is supplied. “Sound” overstates the current evidence.
7. Make the data statement point to a versioned archival release (Zenodo DOI), not
   only a mutable GitHub branch.
8. Add an operator-facing deployment subsection: failure handling, health probing,
   scrape failure fallback, per-model isolation, gateway replication, and what state
   is lost on restart. This directly fits “Practice and Experience.”

## CCPE fit

CCPE is a plausible target if the paper is rewritten as a deployment-and-evidence
paper rather than an algorithmic breakthrough. The cover letter should foreground:

- stock vLLM deployment with no engine fork;
- a real heterogeneous T4/A30 validation;
- negative production evidence about least-loaded and absolute-prediction admission;
- public implementation, raw multi-seed artifacts, and honest operational limits.

Do not quote acceptance rate or a four-day median as established journal facts unless
the journal or Wiley publishes them directly. Aggregator estimates are unreliable,
and a fast first decision often means fast editorial screening, not fast acceptance.

## Journal strategy

There is no reputable journal that simultaneously guarantees high acceptance, very
fast acceptance, Q2/Q3 ranking, and no fees. For hybrid journals, standard
subscription publication normally has no mandatory APC; open access is optional.
Quartiles also vary by year, category, and whether the source is JCR or SJR, so they
must be checked immediately before submission.

Practical shortlist, in order of current manuscript fit:

1. **Concurrency and Computation: Practice and Experience (Wiley)** -- best match if
   reframed around deployability and operational lessons.
2. **International Journal of High Performance Computing Applications (SAGE)** --
   reputable systems venue, but likely expects a more mature performance evaluation.
3. **Journal of Parallel and Distributed Computing (Elsevier)** -- strong reputation
   and relevant scope, but higher novelty/evaluation bar; not a “high acceptance” bet.
4. **Cluster Computing (Springer)** -- topically aligned, but already desk-rejected;
   returning without a substantially different manuscript is not advised.
5. **The Journal of Supercomputing (Springer)** -- scope fit is reasonable, but the
   transfer does not solve the maturity concern by itself.

Before selecting a fallback, verify current indexing/quartile, hybrid status, page or
colour charges, and official decision-time information on the journal site. Do not
optimize for a marketed acceptance percentage; scope fit and a clean editorial pitch
matter more for avoiding another desk rejection.

## Submission recommendation

Do not submit the current Springer PDF unchanged. First produce a clean CCPE PDF with
the cross-SKU result, narrower claims, the legacy pilot removed or demoted, and an
explicit practice/deployment section. That is a meaningful revision; changing the
template alone is not.
