# DIO Launch Playbook

Sequenced plan to take `nisaral/DIO` from "a repo only the authors know about" to
"a repo strangers find, try, and star". Days are relative to **Day 0 = the moment
the repo passes the pre-flight checklist below**.

Everything here is grounded in the actual repository. Where a number is quoted, it
comes from a committed artifact and the path is given so anyone can check it.

---

## 0. Pre-flight (do NOT launch before this is true)

These are the things a stranger hits in the first 90 seconds. Every channel below
sends traffic straight at them, so fix them first.

| # | Gate | Why it costs stars when missing | Status |
|---|------|--------------------------------|--------|
| 1 | `pytest` runs on a clean clone (`pip install -e ".[dev]"` then `pytest tests/ -v`) | The README badge claims "35 passing". If a visitor's clone errors during collection, the badge reads as a lie. The async tests need `pytest-asyncio` present in the dev extra. | verify |
| 2 | Apache-2.0 `LICENSE` visible at the repo root | "Can I use this at work?" is the first question from anyone senior. `dio-serve/LICENSE` exists; the root copy is what GitHub's license detector and the sidebar read. | verify |
| 3 | CI workflow runs the test suite on every push | A green check next to the commit is the cheapest possible trust signal. | verify |
| 4 | Container image (`Dockerfile` + compose) for `dio-serve` | Self-hosters will not `pip install -e` a git checkout on a GPU box if an image exists. | verify |
| 5 | `dio demo --duration 15` works on a cold laptop with no GPU, no Ollama, no vLLM | This is the single best conversion mechanic in the repo. It must not depend on anything the visitor does not have. | verify |
| 6 | README has a visual above the fold (GIF or terminal capture) | Text-only OSS tools get scrolled past. See `docs/launch/REPO_METADATA.md`. | todo |
| 7 | Social preview image uploaded | Controls every link unfurl on HN, Reddit, Slack and Discord. Free reach. | todo |

Do not open any outreach channel (Day 1 onward) until gates 1-5 are green.

---

## Day 0 - repo surface (60-90 minutes, mechanical)

Goal: the repo *looks* like a product before anyone is told about it.
1. Run `docs/launch/gh_repo_setup.ps1` (applies description, topics, homepage,
   issues, discussions). It is re-runnable and only changes what differs.
2. Set the social preview image (Settings -> General -> Social preview).
3. Pin the repo on the `nisaral` profile (Profile -> Customize your pins).
4. Add the one-line positioning at the top of the README, taken from
   `docs/launch/REPO_METADATA.md`.
5. Cut a tagged release (`v0.4.2` is already published):
   - Tag it on the current `main`.
   - Tick "This is a pre-release" until the pre-flight gates are green.
   - Paste release notes from the existing `RELEASE_NOTES.md` / `CHANGELOG.md`.
   - Attach nothing you have not verified. A release with a broken asset is worse
     than a bare tagged release.
6. Confirm the Zenodo DOI resolves and that `.zenodo.json` still links the repo
   (it already does).

Exit criteria: topics visible, description correct, homepage = DOI, social preview
renders, release tagged.

---

## Day 1 - soft launch (three channels, staggered)

Order matters. Start with the smallest, most technical room, fix whatever it
reveals, then scale reach. Posting everywhere at once burns your one shot per
channel on an untested pitch.

| Time (local) | Channel | Post | Why this order |
|---|---|---|---|
| 09:00 | Lobsters (`show` tag) | Short technical framing | Smallest audience, highest signal. Catches factual errors before HN sees them. |
| 13:00 | r/LocalLLaMA | `ANNOUNCEMENTS.md` section B | The core audience for "wrap my vLLM/Ollama fleet". Forgiving and technical. |
| 18:00 | X / Twitter | `ANNOUNCEMENTS.md` section D | Announcement plus GIF. Feeds the later blog and newsletter cycle. |

Rules for Day 1:

- **Never post to more than one channel in the same hour.** Stagger by >=4h.
  Cross-posting identical text within minutes is the fastest way to be read as
  spam.
- **Be present in the comments for the first 6 hours** after each post. Answer
  every technical question. A creator replying within minutes is what turns a
  thread into a conversation instead of an advert.
- **Post as yourself and disclose that you are the author.** Every community
  listed here punishes astroturfing and rewards disclosed authorship.

---

## Week 1 - the main event

| Day | Channel | Post | Notes |
|---|---|---|---|
| Day 2-3 | Hacker News - **Show HN** | `ANNOUNCEMENTS.md` section A | Tue-Thu, 06:00-09:00 US-Pacific. See timing below. |
| Day 4 | r/selfhosted | `ANNOUNCEMENTS.md` section C | Framed as "one endpoint for OpenWebUI/Continue.dev", not as a research paper. |
| Day 4 | Awesome-list PRs | `docs/launch/AWESOME_LISTS.md` | PRs are slow-burn. Open them early; they compound for months. |
| Day 5 | dev.to cross-post | `ANNOUNCEMENTS.md` section F | Blog version of the HN post, canonical link back to the repo. |
| Day 5-6 | Discord communities | Short intro plus real `dio demo` output | Go where the engines live, not where the stars are. |
| Day 6 | LinkedIn | `ANNOUNCEMENTS.md` section E | Different audience (platform / ML-ops). Point at `dio-serve/docs/USE_CASES.md`. |
### Show HN timing

- Submit 06:00-09:00 **US Pacific**, Tuesday to Thursday. That is when the front
  page has the most voting traffic from the audience that cares.
- Submit the **repo URL**, not a blog post. HN reads blog posts as marketing.
- Title is in `ANNOUNCEMENTS.md`. Never ask for upvotes, anywhere, ever.
- Do not resubmit if it dies. One attempt per release.

### What "be present" means concretely

- Keep a terminal ready with `dio demo` output and the test suite so you can
  answer "does it actually work" with a real log instead of a claim.
- Keep the DOI and `dio-serve/results_reanalysis/REPORT.md` in your clipboard for
  anyone who asks for evidence.
- Answer "why not just Nginx round-robin?" head-on every single time. It is the
  one question every thread will ask. The honest answer is in
  `dio-serve/docs/USE_CASES.md`: round-robin treats replicas as interchangeable,
  while queue depth, KV-cache pressure and per-replica slowdowns diverge.

---

## Week 2 - distribution and directories

Now convert one-time reach into durable discovery surfaces.

| Target | Action |
|---|---|
| Awesome lists | Follow up on the PRs opened in Week 1. Nudge politely after 7 days of silence. |
| LibHunt / AlternativeTo / OpenAlternative | Submit (exact mechanics in `docs/launch/AWESOME_LISTS.md`). These become permanent landing pages. |
| Product Hunt | Schedule a launch. Needs a tagline, gallery and a maker who can reply all day. |
| Newsletters | Pitch `PyCoder's Weekly`, `console.dev`, `MLOps.community` and LLM-infra newsletters. Use the dev.to post as the pitch link. |
| Hugging Face | Optional: a small Space that runs `dio demo` against mock backends so "try it" needs no install. |
| Podcasts / Substacks | Pitch `Latent Space` and `Interconnects`. Angle: control-plane research that is also a pip-installable product. |

### Assets to prepare before Week 2

Directory-site visitors bounce without screenshots. Before Week 2 you need:

- 3-5 real screenshots: `dio init` discovery output, the `dio config` routing
  table, a `/debug/metrics` JSON response, and an OpenWebUI chat pointed at
  `:8085`.
- One GIF of `dio demo` showing the NLMS scores rearranging under load.
- A tagline that is a complete phrase, not a fragment. See `REPO_METADATA.md`.

---

## Month 1 - compounding

| Play | Why it matters |
|---|---|
| Publish a measured comparison against Nginx round-robin in the README, with the script that produced it | The only content that earns stars long after launch day is a reproducible number. |
| Ship `v0.4.2` with the top 5 launch issues fixed, and reply in the original threads | Proves the project is alive. "Dead repo" is the number one reason people do not star. |
| Add `good first issue` labels and a `CONTRIBUTING.md` | Turns lurkers into contributors, and contributor PRs are themselves distribution. |
| Write a "How DIO routes" deep dive | HN and Reddit reward second-order technical content from the same author. Link the repo once, at the bottom. |
| Re-run the launch loop at `v0.5.0` | Each release is a legitimate new Show HN. Do not do this more than once a quarter. |
---

## Channel reference: the one-line hook per audience

Full drafts are in `ANNOUNCEMENTS.md`. Use this table so you never paste the wrong
framing into the wrong room.

| Channel | Audience | Hook |
|---|---|---|
| Show HN | Engineers who will actually run it | "A predictive load balancer for self-hosted LLM servers; no engine patches, one URL, Apache-2.0 beta." |
| r/LocalLLaMA | People running vLLM/Ollama on their own GPUs | "I got tired of round-robin wasting my faster GPU, so I made the router learn each backend's latency." |
| r/selfhosted | People running OpenWebUI / home servers | "Point OpenWebUI and Continue.dev at one endpoint; DIO fans out to Ollama and vLLM behind it." |
| Lobsters | Skeptical systems people | "Online NLMS learner in front of OpenAI-compatible servers, plus admission that returns 503 + Retry-After instead of queueing forever." |
| X / Twitter | Practitioners scanning a feed | "One endpoint. Several GPUs. The router learns which one is actually fast." |
| LinkedIn | Platform / ML-ops engineers | "A control plane for multi-instance LLM serving, with a paper, a DOI, and a pip install." |
| dev.to / Hashnode | Python application developers | "A drop-in OpenAI-compatible gateway you can `pip install`." |
| Discord (engine communities) | vLLM / Ollama / MCP users | "Hands-on: `dio init` auto-discovers your engines and writes the config." |
| awesome-list maintainers | Curators | "Apache-2.0, actively maintained, documented, self-hostable." |

---

## Anti-patterns (things that actively kill the launch)

- **Blasting every channel in the same hour.** Reads as a campaign, gets filtered,
  and you lose the ability to iterate between audiences.
- **Asking for upvotes or stars** on HN, Reddit or Discord. Instant ban risk on
  several of these, and it poisons the thread.
- **Posting a benchmark without the script that produced it.** The first comment
  will be "how did you measure this".
- **Claiming production readiness while the docs say "keep `/debug/*` private" and
  auth is opt-in and off by default.** Say beta. Say what is missing. HN rewards that honesty.
- **Ignoring the comments.** A buried reply from the author is worse than no post.
- **Rewriting the post after it takes off.** Typo edits only.

---

## Star-conversion scoreboard

Track these weekly. The launch is only working if the ratios move, not just the
absolute star count.

| Metric | Where |
|---|---|
| Stars | repo |
| Unique visitors and clones | Insights -> Traffic |
| Referrers | Insights -> Traffic -> Referring sites (tells you which channel worked) |
| PyPI downloads (once published) | PyPI stats |
| New issues and discussions | repo |
| awesome-list PRs merged | PR list |

The single most useful signal: **which referrer produced visitors who came back a
second day**. Optimize that channel, not the one with the most upvotes.
