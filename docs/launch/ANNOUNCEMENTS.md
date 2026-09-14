# DIO Announcement Drafts (copy-paste ready)

Six drafts, one per channel. Every claim below is traceable to a file in this
repo; nothing here is invented, and no benchmark number appears without its
provenance and its caveats attached.

**Read this before posting anything:**

- All drafts are plain text in fenced blocks. Copy the block, paste, keep the
  line breaks. Emojis are used sparingly and only where a human would actually
  use them - delete any you do not want.
- Post as yourself and say you are the author. Every community below expects it.
- Do not paste the same text into two channels. The framing differs on purpose;
  reusing one body across subreddits is the classic self-promotion failure.
- Never ask for upvotes or stars.
- Numbers policy: the only quantitative claim used is the committed
  paired-bootstrap reanalysis in `dio-serve/results_reanalysis/REPORT.md`. It is
  *our own* reanalysis of *our own* runs. Always say so. If you add a fresh
  measurement, add the script path next to it.

Shared links:

- Repo: https://github.com/nisaral/DIO
- DOI: https://doi.org/10.5281/zenodo.22085398

---

## A. Hacker News - Show HN

**Title** (79 characters, HN limit is 80):

```
Show HN: DIO - predictive routing for self-hosted LLM fleets (Apache-2.0 beta)
```

**Body** (HN does not render Markdown - the plain-text layout below is intentional;
blank lines become paragraph breaks):

```
I built DIO because the standard answer to "I have two vLLM replicas" is Nginx or
Envoy with round-robin, which assumes the replicas are interchangeable. In
practice they are not: queue depth, KV-cache pressure and transient per-replica
slowdowns diverge, and traffic keeps going to the replica that is currently slow.

DIO is a control plane that sits in front of already-running OpenAI-compatible
servers (vLLM, SGLang, TGI, Ollama) and does four things:

  1. Learns each backend's latency online with a dual-timescale NLMS filter - a
     fast step size for bursts, a slow one for thermal drift. O(1) update per
     request, no background training, no offline model.
  2. Fuses the Prometheus /metrics that vLLM already exposes (KV-cache
     utilisation, waiting queue, prefix-cache hits) into a joint cost.
  3. Routes with model matching and session/prefix affinity for multi-turn chat.
  4. Admits or rejects under overload against a latency SLO, returning
     503 + Retry-After instead of letting the queue grow unbounded.

It speaks both the OpenAI API (/v1/chat/completions, /v1/completions, /v1/models)
and the native Ollama API (/api/chat, /api/generate, /api/tags), so existing
clients work unchanged - including OLLAMA_HOST. There is also an MCP server
(`dio mcp`) exposing tools for Cursor / Claude Desktop / VS Code so an assistant
can query cluster state and preview routing before sending a request.

You can see it work with no GPU and no engines installed: `dio demo` starts mock
backends and shows the learned scores rearranging under load. `dio init` scans
localhost for Ollama (11434), vLLM (8000/8001) and SGLang (30000) and writes a
dio.yaml for whatever it finds.

What it is not: not a model, not a quantizer, not a vLLM fork. It never patches
the engine - it uses the HTTP API and the metrics endpoint you already expose. If
you run a single GPU with light traffic you do not need it; plain vLLM is fine.

Status is v0.4.0, beta, Apache-2.0, Python 3.9+. The honest gaps: no auth or
tenant quotas (keep /debug/* private), and admission is calibration-robust rather
than formally sound. Both are documented.

On evidence, being explicit: the numbers in the repo are our own runs, not a
third-party benchmark. In a committed paired-bootstrap reanalysis (n=10 paired
runs, 100k resamples, fixed seed), hybrid routing vs round-robin showed 40.9%
lower p99, 95% CI [762, 1969] ms, Cohen dz 1.34, 8/10 wins. The file also lists
the caveats, and there is a paper with an artifact DOI behind the design.

Repo: https://github.com/nisaral/DIO
DOI: https://doi.org/10.5281/zenodo.22085398

Happy to answer questions on the scheduler, the NLMS formulation, or how the
measurements were set up - including the parts that did not work.
```

**Posting notes**

- Submit the repo URL, not a blog post.
- Tue-Thu, 06:00-09:00 US-Pacific. Stay in the thread for 6 hours.
- The first critical comment will be about the evaluation methodology. Answer it
  with the file path and the caveat, not with defensiveness.
---

## B. r/LocalLLaMA (long-form technical post)

**Title**

```
DIO: an OpenAI- and Ollama-compatible gateway that learns each backend's latency online and sheds overload (Apache-2.0, v0.4.0 beta)
```

**Body**

```
I run more than one inference server, and I kept hitting the same thing: round-robin
in front of them is fine until the replicas stop being equal. One box gets hotter,
its queue fills, KV cache pressure climbs, and the load balancer happily keeps
sending it a fair share of traffic because it only knows about connection counts.
So I wrote the router I wanted instead.

**What it is**

DIO is a control plane that sits in front of servers you are already running -
vLLM, SGLang, TGI, Ollama, LM Studio, anything that speaks OpenAI-compatible
HTTP. It does not patch the engine, does not fork vLLM, and does not touch your
weights. It talks HTTP to your engines and reads the Prometheus /metrics vLLM
already exposes.

**How the routing works**

1. Online latency learning. Each backend gets a dual-timescale NLMS filter: a
   fast step size that reacts to bursts and a slow one that tracks thermal drift.
   It updates in O(1) per request with no background training step and no
   offline model to retrain.
2. Joint cost. The learned latency is combined with queue depth, KV-cache
   utilisation and prefix-cache hits from the engine's own /metrics, plus tier
   and VRAM headroom. So it is not just "which one answered fastest last time".
3. Affinity. Multi-turn chat gets a bonus for session/prefix reuse, which matters
   a lot when your engine has a prefix cache.
4. Admission. Under overload it compares the predicted cost against a latency SLO
   and returns 503 + Retry-After instead of queueing until everyone's p99 falls
   over. Admission is decoupled from the raw cost - you can also run it in
   rank-only or absolute mode.

**What makes it easy to actually try**

- It speaks both APIs. OpenAI on /v1/chat/completions, /v1/completions,
  /v1/models. Ollama native on /api/chat, /api/generate, /api/tags, /api/version,
  /api/show. You can literally `export OLLAMA_HOST=http://127.0.0.1:8085` and
  keep using the Ollama CLI.
- `dio init` scans localhost for Ollama (11434), vLLM (8000, 8001) and SGLang
  (30000), detects the loaded models, and writes a dio.yaml for you.
- `dio demo` runs a zero-GPU demo with mock backends, so you can watch the scores
  rearrange under load before wiring anything real up.
- `dio bench-smoke` compares NLMS against round-robin on synthetic heterogeneous
  workers.
- There is an MCP server (`dio mcp`) with four tools, so Cursor / Claude Desktop
  / VS Code can query cluster state and preview latency before sending a request.

**Config is one file** (`dio.yaml`): backends with engine type, tier and models,
then scheduler and admission knobs. Strategies available: nlms, rls, ewma,
static, round_robin, least_loaded - which also makes it easy to A/B against the
classic baselines.

**What is in the repo**

The scheduler and gateway are a small Python package (3.9+). There are 89 tests
covering config parsing, multi-model routing, SSE + NDJSON streaming, the Ollama
adapter and the MCP server. License is Apache-2.0.

**Honest status**

This is a v0.4.0 beta. There is no auth and no tenant quotas yet, so keep the
/debug/* endpoints on a private network. The admission gate is calibration-robust
rather than formally sound - it is percentile gating, not a proof of SLO
adherence. All of that is written down in the docs rather than hidden.

The evaluation numbers in the repo are our own runs, not an independent
benchmark. The committed paired-bootstrap reanalysis (n=10 paired runs) shows
hybrid routing vs round-robin at 40.9% lower p99, 95% CI [762, 1969] ms, 8/10
wins - and the same file lists where the result is weaker. Please read it with
that in mind; I would rather you trust the caveats than the headline.

Repo: https://github.com/nisaral/DIO
DOI: https://doi.org/10.5281/zenodo.22085398

Feedback very welcome, especially from anyone running heterogeneous GPUs - that
is the case I most want broken.
```

**Posting notes**

- Best window: weekday morning US time. This sub is forgiving and technical, and
  it is the single highest-value audience for this project.
- Reply to every "how does this compare to X" with specifics, and link
  `dio-serve/docs/USE_CASES.md` for the "do I need this at all" questions.
- If someone reports a bug in the thread, fix it and reply with the commit.
---

## C. r/selfhosted ("one endpoint" framing)

**Title**

```
Point OpenWebUI and Continue.dev at one endpoint - DIO fans out to your Ollama and vLLM servers and routes around the slow one
```

**Body**

```
Most of us end up with more than one thing serving models: Ollama on the box with
the big card, a vLLM container on another, maybe an old GPU that is fine for small
models. Then every client needs to know all the ports, and none of them fail over
or pick sensibly.

DIO puts one endpoint in front of all of them.

**What you actually do**

1. Install it (Python 3.9+): `pip install -e dio-serve` from the repo for now.
2. Run `dio init`. It scans localhost for Ollama on 11434, vLLM on 8000/8001 and
   SGLang on 30000, detects which models are loaded, and writes a `dio.yaml`.
3. Edit `dio.yaml` to add anything remote, like an Ollama box on another machine.
4. Run `dio serve`. It listens on :8085.

Then, depending on the client:

- OpenWebUI: set the Ollama base URL to `http://<host>:8085` (native /api
  endpoints are supported: /api/chat, /api/generate, /api/tags).
- Anything OpenAI-shaped (Continue.dev, LiteLLM, the OpenAI SDK, LangChain): base
  URL `http://<host>:8085/v1`.
- The Ollama CLI: `export OLLAMA_HOST=http://<host>:8085` and use it as normal.

No client changes beyond the URL. Both API surfaces are served from the same
process.

**Why route instead of just round-robin**

Round-robin sends each backend an equal share, which is wrong the moment they are
not equally fast or not equally busy. DIO learns each backend's latency online
and combines that with the engine's own metrics (queue depth, KV-cache pressure
for vLLM) to pick. It also keeps a multi-turn chat on the backend that already has
its prefix cached, which is noticeable on long conversations.

When you are overloaded, it answers `503` with `Retry-After` instead of queueing
requests until everything times out. Clients that retry will do the right thing,
and you stop the pile-up.

**Try it without any GPU first**

`dio demo` starts mock backends and shows the learned routing scores changing
under load. It needs no GPU, no engines and no model downloads, so you can decide
whether it is interesting before touching your real setup.

**Things to know before you deploy it**

- It is a v0.4.0 beta, Apache-2.0.
- There is no authentication or multi-tenant quota yet. Put it behind your
  reverse proxy for anything reachable, and keep the /debug/* endpoints off the
  public network.
- Run one pool per model family if the models have very different latency
  profiles; mixing a long-context model and a tiny model in one learner is not
  recommended and is documented as such.

Repo and docs: https://github.com/nisaral/DIO

If you have a mixed home setup - say a 3090 next to an older card - that is
exactly the case this was built for, and I would like to hear what breaks.
```

**Posting notes**

- This audience is allergic to hype and to "SaaS that is actually open source".
  Lead with the concrete client config, as above, not with the algorithm.
- Expect "why not just use LiteLLM / Traefik / Nginx". Have a real answer ready:
  those balance on request counts or weights, not on per-backend learned latency
  plus engine telemetry.
- Do not post on the same day as the HN attempt.
---

## D. X / Twitter (single post, 276 characters)

Paste this as one post. If you want a thread, this is post 1 and the repo link is
post 2 - but the single post below already stands alone, which is what you want
because quote-posting your own link is noise.

```
One endpoint. Several GPUs. The router learns which one is actually fast.

DIO: predictive LLM gateway — NLMS latency learning + SLO admission in front of vLLM/Ollama/SGLang/TGI. OpenAI + Ollama APIs. MCP server for your IDE. Apache-2.0, v0.4.0 beta.

github.com/nisaral/DIO
```

**Posting notes**

- 276/280 characters, verified. Do not add a hashtag; it would push it over and
  hashtags do nothing in this niche.
- Attach the `dio demo` GIF. A post with the product moving outperforms the same
  text alone by a wide margin.
- Post it the day after the subreddit post, not the same hour.

---

## E. LinkedIn

**Body**

```
We just published v0.4.0 of DIO, an open-source control plane for teams running
more than one LLM inference server.

The problem it addresses is mundane and expensive: once you have several vLLM,
SGLang, TGI or Ollama instances, the usual setup is a load balancer doing
round-robin. That assumes the replicas are interchangeable. They are not. Queue
depth, KV-cache pressure and transient per-replica slowdowns diverge, and traffic
keeps landing on whichever replica is currently struggling. The result shows up
as p99 latency that no amount of GPU budget fixes.

DIO learns each backend's latency online - a dual-timescale NLMS filter with O(1)
updates and no offline training - and combines that with the telemetry the engines
already expose to route each request. Under overload it rejects with 503 and
Retry-After rather than letting the queue grow past the latency objective.

What makes it practical to adopt:

- It is non-invasive. You keep your engines exactly as they are; DIO only needs
  their HTTP endpoints.
- It exposes both the OpenAI API and the native Ollama API, so existing
  applications, OpenWebUI, Continue.dev and LangChain integrations only need a
  base-URL change.
- It ships with an MCP server, so AI coding assistants can inspect cluster state
  and preview routing before sending a request.
- Configuration is a single YAML file, and `dio init` discovers local engines and
  writes it for you.
- Apache-2.0, Python 3.9+, with a test suite covering the gateway, streaming,
  routing, the Ollama adapter and the MCP server.

The design is backed by a research artifact with a DOI, which is also the source
of the evaluation numbers - we are explicit that they are our own runs rather
than an independent benchmark, and the committed reanalysis lists its own
caveats.

It is a beta: there is no authentication or multi-tenant quota yet, and the
admission gate is percentile-based rather than formally verified. Both are
documented rather than glossed over, and feedback from teams running
heterogeneous GPU fleets is exactly what we want next.

Repo: https://github.com/nisaral/DIO
Artifact DOI: https://doi.org/10.5281/zenodo.22085398

#LLM #MLOps #OpenSource #AIInfrastructure #vLLM
```

**Posting notes**

- LinkedIn renders the first 2-3 lines before "see more". The first two
  paragraphs above are written for that fold.
- Attach the architecture diagram from `dio-serve/docs/` or a screenshot of the
  `dio config` routing table. LinkedIn posts with an image get materially more
  reach.
- Do not paste the X draft here; the audience is different.
---

## F. dev.to / Hashnode / blog skeleton

Cross-post the same article to dev.to and Hashnode with a canonical link back to
your own site (or to the dev.to post) so the SEO does not split. Target length
1200-1600 words. Headings below are the actual structure.

```markdown
# Why round-robin is the wrong load balancer for your LLM servers

## The setup everyone ends up with

Two paragraphs: N vLLM/Ollama/SGLang instances behind Nginx/Envoy with
round-robin. Why that is the default answer, and why it is fine until it is not.
Name the failure: one replica gets hot, its queue deepens, KV cache pressure
rises, and round-robin keeps feeding it because it only counts connections.

## What "the replicas are not interchangeable" actually means

Short technical narrative. Three concrete divergences:
- thermal throttling changes effective tokens/sec over minutes
- KV-cache pressure changes admission latency at the engine, not the network
- prefix-cache hits mean the same request is cheap on one replica and expensive
  on another
Do not invent numbers here; describe the mechanism.

## The idea: learn the backend, then route

Introduce online latency learning with a dual-timescale NLMS filter. Explain the
two step sizes as "one that reacts to bursts, one that tracks slow drift". Make
the O(1) update point - no background training job, no offline model to retrain,
no drift between the model and reality.

## Adding engine telemetry to the decision

Show how the engine's own /metrics (queue depth, KV-cache utilisation,
prefix-cache hits) enters a joint cost alongside the learned latency, plus tier
and VRAM headroom. Emphasise that this is non-invasive: the metrics endpoint
already exists.

## Admission, not just routing

Explain why routing alone does not save your p99 once the whole fleet is
oversubscribed, and why 503 + Retry-After beats an unbounded queue. Show the
admission modes (empirical percentile gate, rank-only, absolute).

## The boring parts that make it usable

- Both APIs: OpenAI and native Ollama on the same port.
- `dio init` auto-discovery of Ollama/vLLM/SGLang and generated dio.yaml.
- `dio demo` zero-GPU demo so nobody has to trust a blog post.
- MCP server so an IDE assistant can query the cluster.

## Try it in four commands

Fenced block: clone, install, `dio init`, `dio serve` - plus `dio demo` as the
no-GPU path.

## What it is not, and what is missing

Be explicit: not a model, not a quantizer, not a vLLM fork; no auth yet; keep
/debug/* private; admission is calibration-robust, not formally sound. This
section is why technical readers will trust the rest of the post.

## Where the numbers come from

Point at the committed reanalysis and repeat the caveat that these are the
authors' own runs, with n and confidence intervals, not a third-party benchmark.

## Links

Repo, DOI, docs paths.
```

**Posting notes**

- dev.to tags: `llm`, `ai`, `devops`, `opensource` (four is the practical max
  before the feed ignores the rest).
- Add the canonical URL if you post it on two platforms.
- The article's job is not to explain everything - it is to make someone open a
  terminal. Link the repo three times: once in the intro, once after "Try it",
  once at the end.

---

## Appendix: fact sheet for comment replies

Keep this open while the threads are live. Every item is checkable in the repo.

| Claim | Where it comes from |
|---|---|
| Wraps vLLM, SGLang, TGI, Ollama, any OpenAI-compatible HTTP server | `dio-serve/docs/PERFORMANCE.md` (supported engines table) |
| Dual-timescale NLMS, O(1) updates, no background training | `dio-serve/README.md`; `dio-serve/src/dio/scheduler.py` |
| Both OpenAI and native Ollama APIs on the same port | `dio-serve/src/dio/gateway.py`; `dio-serve/README.md` |
| `/api/chat`, `/api/generate`, `/api/tags`, `/api/version`, `/api/show` | `dio-serve/README.md`; `dio-serve/tests/test_ollama_adapter.py` |
| `dio init` discovers Ollama 11434, vLLM 8000/8001, SGLang 30000 | `dio-serve/src/dio/config_file.py`; README quick start |
| Admission returns 503 + Retry-After | `dio-serve/docs/PRODUCTION.md`; README |
| MCP server with 4 tools over JSON-RPC 2.0 stdio | `dio-serve/src/dio/mcp.py`; `dio-serve/tests/test_mcp.py` |
| Strategies: nlms, rls, ewma, static, round_robin, least_loaded | `dio-serve/src/dio/config.py`; `dio.example.yaml` |
| 89 tests | count of `def test_` across `dio-serve/tests/` |
| Apache-2.0 | `dio-serve/LICENSE`; `pyproject.toml` |
| Python 3.9+ | `pyproject.toml` (`requires-python`) |
| Artifact DOI | `CITATION.cff`, `.zenodo.json` |
| 40.9% p99 improvement vs round-robin, n=10, 95% CI [762, 1969] ms, dz 1.34, 8/10 wins | `dio-serve/results_reanalysis/REPORT.md` (authors' own paired-bootstrap reanalysis) |
| No auth / no tenant quotas; keep `/debug/*` private | `dio-serve/docs/PRODUCTION.md` |
| Admission is calibration-robust, not formally sound | `RELEASE_NOTES.md` (v0.3.0-rc1 entry) |

**Answers to the three questions you will get every time**

1. *"Why not just Nginx/Envoy round-robin?"*
   Round-robin is stateless with respect to backend performance. DIO learns
   per-backend latency online and folds in the engine's own queue and KV-cache
   metrics, so it can move traffic off a replica that is currently slow. If your
   replicas are genuinely identical and your traffic is flat, you do not need it.

2. *"Is this just LiteLLM / a proxy?"*
   LiteLLM-style proxies route by model and provider with static weights and
   retries. DIO's decision is a measured, continuously updated cost per backend,
   plus an admission gate that can reject rather than queue. They are
   complementary: DIO can sit behind a model-name router.

3. *"What is your evaluation?"*
   Our own committed runs, reanalysed with paired bootstrap. Here is the file,
   here is n, here are the confidence intervals, and here are the caveats. We are
   not claiming an independent benchmark.
