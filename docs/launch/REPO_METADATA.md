# GitHub Repo Metadata - exact settings to apply

Copy-paste values for `nisaral/DIO`. `docs/launch/gh_repo_setup.ps1` applies the
machine-settable parts (description, topics, homepage, issues, discussions); the
image and README items are manual by nature.

**Applied on 2026-09-14** (run from `docs/launch/gh_repo_setup.ps1`): description,
DOI homepage, 20 topics and Discussions are live. The script treats the topic list
as the desired *final* set -- it removes topics that are not in the list, because
GitHub caps topics at 20 and the repo already carried six loosely-related ones
(`distributed-systems`, `inference`, `loadbalancing`, `nlms-algorithm`,
`research-artifact`, `slo`). Re-running it is a no-op.

Still manual: the social-preview upload, the profile pin, and the announcement
posts in `ANNOUNCEMENTS.md`.

---

## 1. Repository description (the "About" field)

**Hard limit 160 characters.** This one is 150:

```
DIO: non-invasive LLM gateway that learns per-backend latency online (dual-timescale NLMS) and routes vLLM/Ollama/SGLang/TGI with SLO-aware admission.
```

Why this shape: it names the category ("LLM gateway"), the differentiator
("learns per-backend latency online", "dual-timescale NLMS") and the engines
people actually search for ("vLLM/Ollama/SGLang/TGI"). GitHub search indexes this
field, so the engine names are doing real work.

Alternates if you want to test tone (all under 160):

- `One OpenAI-compatible endpoint in front of many vLLM/Ollama/SGLang/TGI servers. Learns which backend is fast and sheds overload. Apache-2.0.` (140)
- `Predictive load balancer for self-hosted LLM fleets. Learns latency online, routes on a joint cost, rejects overload with 503 + Retry-After.` (141)

---

## 2. Topics (GitHub allows max 20)

Ordered by search value. Add them in this order; if the UI truncates, the first
ones are the ones that matter.

| # | Topic | Why |
|---|-------|-----|
| 1 | `llm` | Broadest entry point |
| 2 | `llm-serving` | Exact niche |
| 3 | `llm-inference` | Exact niche, high traffic |
| 4 | `llm-gateway` | Category term people search for gateways |
| 5 | `llmops` | Practitioner topic |
| 6 | `mlops` | Adjacent, larger |
| 7 | `vllm` | Engine name, high intent |
| 8 | `ollama` | Engine name, very high traffic |
| 9 | `sglang` | Engine name |
| 10 | `openai-api` | Wire-format searchers |
| 11 | `openai-compatible` | Wire-format searchers |
| 12 | `model-serving` | Classic MLOps term |
| 13 | `inference-server` | Classic serving term |
| 14 | `load-balancing` | The problem statement |
| 15 | `admission-control` | The differentiator |
| 16 | `model-context-protocol` | MCP angle, small but trending |
| 17 | `mcp-server` | MCP angle, trending fast |
| 18 | `ai-infrastructure` | Category |
| 19 | `self-hosted` | Very large topic, right audience |
| 20 | `python` | Largest topic, language filter |

`self-hosted` and `python` are the two "big bucket" topics: they will not make
you discoverable on their own, but they put you in the filtered feeds people
actually browse.

---

## 3. Homepage URL

Set the About homepage to the artifact DOI:

```
https://doi.org/10.5281/zenodo.22085398
```

Rationale: it is the strongest credibility signal the project has, it is
stable, and it makes the paper one click away from the repo header. If you would
rather drive traffic to the docs, use
`https://github.com/nisaral/DIO/tree/main/dio-serve` - but the DOI is the better
choice for an OSS launch.
---

## 4. Social preview image

Settings -> General -> Social preview -> Upload image.

- **Size:** 1280 x 640 px (GitHub's documented ratio, 2:1).
- **File:** PNG under 1 MB. Do not use the 140 px `docs/assets/logo.jpg` as-is;
  it will look mushy.
- **Composition that works for engineering tools:**
  - Left two thirds: the wordmark `DIO` plus the one-line positioning.
  - Right third: a real terminal snippet, monospaced, showing the router
    choosing between two backends. Real output beats illustration for this
    audience.
  - Bottom strip: `Apache-2.0` and `v0.4.0 beta`.
- **Do not** put a sentence longer than ~12 words in the image. It is rendered at
  thumbnail size in Twitter/X, Slack, Discord and Reddit.

Suggested single line for the image:

```
DIO - predictive routing for self-hosted LLM fleets (vLLM / Ollama / SGLang / TGI)
```

Keep the source file (`docs/launch/assets/social_preview.png` or similar) in the
repo so it can be regenerated; only the uploaded copy matters to GitHub.

---

## 5. README changes that convert visitors into stars

These are recommendations for whoever owns the README files. They are listed
here because they are launch-critical, not because this document edits them.

### 5.1 Above the fold: a 30-second slot

Directly under the badges, add a single line and an asset:

```markdown
<p align="center">
  <img src="docs/assets/demo.gif" alt="dio demo: NLMS scores rearranging under load" width="720"/>
</p>
```

The GIF must show, in under 30 seconds: `dio init` discovering engines, then
`dio serve` (or `dio demo`) with the per-backend scores visibly changing. A brand
animation will not do; show the product working. This is the highest-value single
change on this list.

### 5.2 The "why not just Nginx round-robin?" hook

Put this immediately after the first code block, as plain text, not a
sub-heading buried lower:

> **Why not just Nginx round-robin in front of N vLLM replicas?**
> Round-robin assumes every replica is equally fast at every moment. In practice
> queue depth, KV-cache pressure and transient per-replica slowdowns diverge. DIO
> learns each backend's latency online (dual-timescale NLMS, O(1) updates) and
> routes on a joint cost, then rejects overload with `503 + Retry-After` instead
> of letting the queue grow past your SLO.

This pre-empts the single most common objection and is the sentence that turns a
scanner into a reader.

### 5.3 Badges worth having (and the ones that are noise)

Keep the existing four badges (version, python, tests, license). Add only:

- `DOI` badge -> `https://doi.org/10.5281/zenodo.22085398` (use a static
  shields.io badge; Zenodo's own badge is fine too). For a research-backed tool
  this is a genuine differentiator.
- `PyPI` badge **only once the package is actually published.** A badge linking
  to a non-existent package is worse than no badge.

Skip: star-count badges, "made with love", contributor-count for a young repo.

### 5.4 A 20-second "try it" path

The first code block should stay exactly as short as it is now (clone, install,
`dio init`, `dio serve`, plus `dio demo`). Do not add steps. If anything, cut the
`git clone` line by mentioning `pip install dio-serve` once that is live.
### 5.5 Repository features to switch on

| Feature | Setting | Why |
|---|---|---|
| Issues | On | Launch traffic generates questions; a repo with issues off looks abandoned. |
| Discussions | On | Absorbs "how do I..." traffic that would otherwise become low-quality issues. Create a `Show your setup` discussion at launch. |
| Wiki | Off (unless populated) | An empty wiki is a dead end. Docs live in `dio-serve/docs/`. |
| Projects | Off | Empty project boards add clutter. |
| Sponsors / funding | Your call | Not needed for credibility in this niche. |
| Topics | Set (see section 2) | Primary GitHub discovery surface. |

### 5.6 Release hygiene at launch

- Tag `v0.4.0` and mark it **pre-release** until CI is green and `dio demo` is
  verified on a clean machine.
- Release body: paste from `RELEASE_NOTES.md`, then append a `## Try it` block
  with the four-line quick start.
- Do not attach build artifacts you have not opened and tested.

---

## 6. Apply checklist

| Item | How | Done by |
|---|---|---|
| Description | `docs/launch/gh_repo_setup.ps1` | script |
| Topics (20) | `docs/launch/gh_repo_setup.ps1` | script |
| Homepage = DOI | `docs/launch/gh_repo_setup.ps1` | script |
| Issues on | `docs/launch/gh_repo_setup.ps1` | script |
| Discussions on | `docs/launch/gh_repo_setup.ps1` | script |
| Social preview image | Settings -> General -> Social preview | manual |
| Pinned repo | Profile -> Customize your pins | manual |
| README demo GIF | README owner | manual |
| "Why not round-robin" paragraph | README owner | manual |
| Zenodo/DOI badge | README owner | manual |
| `v0.4.0` pre-release | Releases -> Draft a new release | manual |
