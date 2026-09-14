# Awesome Lists and Directories - submission targets

Where to submit DIO to get durable, compounding discovery. Awesome-list entries
sit in front of exactly the audience that searches for this category, and
directory pages rank in Google for "<competitor> alternatives" queries for years.

**Important honesty note about verification.** The list below was assembled from
established knowledge of these projects, but the machine this document was
written on had no outbound network access, so the URLs could not be fetched and
confirmed live. Every entry therefore carries a `Confidence` value, and section 1
gives you a one-command way to verify all of them before you spend time on a PR.
Treat "Medium" rows as leads, not as facts.

---

## 0. Verify everything first (30 seconds, one command)

Run this from the repo root with an authenticated `gh` CLI (or swap in `curl -sI`
if you prefer). It prints `OK` or `MISSING` per repo:

```powershell
$targets = @(
  'awesome-selfhosted/awesome-selfhosted',
  'punkpeye/awesome-mcp-servers',
  'modelcontextprotocol/servers',
  'InftyAI/Awesome-LLMOps',
  'tensorchord/Awesome-LLMOps',
  'Hannibal046/Awesome-LLM',
  'kyrolabs/awesome-langchain',
  'e2b-dev/awesome-ai-agents',
  'steven2358/awesome-generative-ai',
  'ml-tooling/best-of-ml-python'
)
foreach ($t in $targets) {
  if (gh repo view $t --json name 2>$null) { Write-Host "OK      $t" }
  else { Write-Host "MISSING $t" }
}
```

If a repo is missing or has been renamed, search for the current name before
giving up on the category - these lists get renamed and re-homed often.

Then re-read the target's most recent `CONTRIBUTING.md` before opening a PR. The
mechanics below are the *shape* of the process; the maintainers own the details
and they change.

---

## 1. Awesome lists, ranked by expected value

| # | Repository | Section to use | Fit | Confidence |
|---|---|---|---|---|
| 1 | `punkpeye/awesome-mcp-servers` | Developer tools / infrastructure category | Very high - DIO ships a real MCP server | High |
| 2 | `awesome-selfhosted/awesome-selfhosted` | Software -> Generative AI | High - one-line install, self-hostable, OSS | High |
| 3 | `tensorchord/Awesome-LLMOps` | Serving / inference | High - core category match | High |
| 4 | `InftyAI/Awesome-LLMOps` | Serving / LLM gateway | High | High |
| 5 | `Hannibal046/Awesome-LLM` | Inference / serving systems | Medium-high | High |
| 6 | `ml-tooling/best-of-ml-python` | Model Serving / Inference | High but has a stars threshold | High |
| 7 | `kyrolabs/awesome-langchain` | Tools / serving | Medium - LangChain base_url angle | High |
| 8 | `steven2358/awesome-generative-ai` | Infrastructure / serving | Medium | High |
| 9 | `e2b-dev/awesome-ai-agents` | Infrastructure | Low-medium | Medium |
| 10 | `modelcontextprotocol/servers` | No longer a community list - see section 2 | High, but different mechanic | High |

Rule of thumb: submit to 1, 2 and 3 first. Those three reach the people who run
the GPUs. The rest are volume.
---

## 2. Per-target detail

### 2.1 `punkpeye/awesome-mcp-servers` (highest-value MCP target)

- **Section:** choose the closest existing category. An infrastructure/dev-facing
  server like this belongs with the developer-tooling category, not with
  databases or knowledge bases. Read the current category list in the README and
  pick the one whose neighbours are similar dev tools.
- **Entry format:** this list uses emoji markers after the link to encode
  language, local/cloud scope and supported operating systems. The shape is:

  ```markdown
  - [dio-serve](https://github.com/nisaral/DIO) 🐍 🏠 🍎 🪟 🐧 - MCP server exposing live LLM routing telemetry: list models, predict per-backend latency, and route prompts across a vLLM/Ollama/SGLang/TGI fleet.
  ```

  Copy the exact marker set from neighbouring entries rather than trusting this
  line - the list's legend is authoritative.
- **Contributing rules:** one entry per server, appended in the right category,
  links must resolve, and the server must genuinely implement MCP. Their CI
  validates the README, so a malformed line fails the PR check.
- **Mechanic:** fork, add one line, open a small PR titled something like
  `Add dio-serve MCP server`. Small single-purpose PRs get merged fastest.
- **Bonus:** once listed, the same entry is mirrored by several MCP aggregators
  that crawl this list, so one merged PR produces several listings.

### 2.2 `awesome-selfhosted/awesome-selfhosted`

- **Section:** Software -> Generative AI (verify the current category name in the
  README; categories get reorganised).
- **Entry format (exact shape):**

  ```markdown
  - [DIO](https://github.com/nisaral/DIO) - Predictive gateway that learns each backend's latency and routes LLM requests across vLLM, Ollama, SGLang and TGI instances. ([Source Code](https://github.com/nisaral/DIO)) `Apache-2.0` `Python`
  ```

- **Contributing rules (these are enforced, not advisory):**
  - The project must be free software. Apache-2.0 is fine; a "source available"
    or open-core license is not.
  - The description must be a single plain sentence, end with a period, and avoid
    marketing language.
  - The entry must include a `Source Code` link and the license + language tags
    in backticks, matching the surrounding formatting exactly.
  - Entries are alphabetical within a section.
  - Their CI runs an awesome-lint pass; an unformatted line fails the build. Run
    their linter locally before pushing if the repo documents a command for it.
- **Mechanic:** README-only PR. Read their `CONTRIBUTING.md` first - this is one
  of the strictest lists on the web and a low-effort PR gets closed.

### 2.3 `tensorchord/Awesome-LLMOps`

- **Section:** the serving / inference section (the list groups engines and
  serving infrastructure there).
- **Entry format:** markdown table row - copy the existing row shape exactly. The
  description column is short and factual, not a pitch.
- **Contributing rules:** maintainers want real, maintained projects; they
  generally reject anything that has not shipped in the last year.
- **Mechanic:** one-row PR.

### 2.4 `InftyAI/Awesome-LLMOps`

- **Section:** the gateway / serving category (this list explicitly tracks LLM
  gateways and serving infrastructure, which is exactly DIO's category).
- **Entry format:** match the neighbouring rows in the same table or bullet list.
- **Contributing rules:** they ask for the project link plus a one-line
  description; keep it under a line.
- **Mechanic:** one-row PR.

### 2.5 `Hannibal046/Awesome-LLM`

- **Section:** the inference / serving systems list.
- **Entry format:** bullet with link plus short description, matching the
  alphabetical or topical ordering used in that subsection.
- **Contributing rules:** broad and popular list; PRs are merged slowly, so treat
  this as a month-1 item rather than a launch-day one.

### 2.6 `ml-tooling/best-of-ml-python`

- **Section:** Model Serving / Inference category.
- **Entry format:** this list is generated. You do not edit the README - you add
  an entry to `projects.yaml`:

  ```yaml
  - name: DIO Serve
    github: nisaral/DIO
    description: Predictive NLMS orchestrator and universal LLM gateway that routes across vLLM, Ollama, SGLang and TGI.
    tags:
      - model-serving
      - inference
  ```

  Then regenerate the README with their generator (the repo documents the command,
  usually `make`) so the diff includes both files.
- **Contributing rules:** projects must be open source, actively maintained, and
  above their minimum star threshold. That threshold is the catch - do this one
  **after** the launch has put stars on the repo, not before. A submission that
  fails the threshold wastes a maintainer's time and yours.
- **Mechanic:** PR containing the `projects.yaml` entry plus the regenerated
  README.
### 2.7 `kyrolabs/awesome-langchain`

- **Section:** the tooling / serving section.
- **Fit:** medium but non-obvious value. LangChain users change `base_url` and
  everything works, which is a genuine integration story worth one line.
- **Entry format:** `- [Name](url): short description.` - copy the surrounding
  entries.
- **Contributing rules:** alphabetical or topical placement as used in that list;
  one line; no sub-bullets.

### 2.8 `steven2358/awesome-generative-ai`

- **Section:** infrastructure / serving (verify the current heading).
- **Entry format:** matching bullet style, link first.
- **Fit:** medium. Broad audience, lower intent than the LLMOps lists.

### 2.9 `e2b-dev/awesome-ai-agents`

- **Fit:** low-medium. Only worth it if the MCP angle is your headline, since
  agents and MCP go together. Submit after the MCP list, not instead of it.

---

## 3. MCP-specific directories and registries (do not skip these)

The MCP ecosystem has its own distribution layer, and it is far less crowded than
the awesome lists. This is the highest-leverage section for DIO's MCP server.

| Target | What it is | Mechanic | Confidence |
|---|---|---|---|
| Official MCP Registry (`registry.modelcontextprotocol.io`) | The canonical registry from the MCP project | Publish a `server.json` manifest for the stdio server, authenticate with GitHub, then publish with the registry's CLI | Medium-high - the registry was in preview and its tooling changed during 2025, so read their current publishing docs |
| `modelcontextprotocol/servers` | The official repo | It no longer maintains the community server list; it points readers at the community list and the registry. Do not open a PR adding a row - publish to the registry and get listed through the community list instead | High |
| Smithery (`smithery.ai`) | MCP server directory used by several clients | Publish via their site or CLI; local stdio Python servers are supported, and a small config file describes how to launch the server | Medium-high |
| Glama (`glama.ai/mcp/servers`) | MCP server index with claim/verification flow | Wait for it to crawl the repo, then claim the listing and fill in details | Medium |
| PulseMCP (`pulsemcp.com`) | MCP directory plus a widely read newsletter | Submit through their form; the newsletter mention is the real prize | Medium-high |
| `mcp.so` | Community MCP directory | Submit through the site form | Medium |
| Continue Hub (`hub.continue.dev`) | IDE-assistant hub that surfaces MCP servers | Publish a hub block for the server so Continue users can add it in one click | Medium |

Because several of these crawlers index the same sources (the community list and
the official registry), do those two first and let the aggregators pick the entry
up. Verify each listing afterwards and correct the description where the crawler
got it wrong.

---

## 4. General software directories

These are not niche, but they rank in search and they are permanent pages.

| Target | Mechanic | Notes |
|---|---|---|
| AlternativeTo (`alternativeto.net`) | Create an account, use "Add an app", fill description, license, platforms, screenshots. Then list it as an alternative to the tools people compare against: vLLM, Ollama, LiteLLM, Nginx, Traefik, OpenRouter. | Manual review. Do not ask anyone to vote - vote manipulation gets listings removed. Self-submission is allowed; astroturfing is not. |
| LibHunt (`libhunt.com`) | Search for the repo; if it exists, claim it as the owner; if not, submit it. Then add the LibHunt badge to the README if you want the listing kept fresh. | Auto-indexes repos past a star threshold, so this one unlocks itself after launch. |
| OpenAlternative (`openalternative.co`) | Submit through their site's submit flow with the repo URL, license and a short description. | Open-source-only directory, so Apache-2.0 fits cleanly. Paid featured placement exists; free listing is enough. |
| Product Hunt | "Ship" flow: tagline (<=60 chars), gallery images, topics, scheduled launch date. | Needs a real gallery and someone replying all day. Do this in Week 2, never the same week as HN. |
| DevHunt (`devhunt.org`) | Product-Hunt-style weekly launch for dev tools | Better audience-to-noise ratio than Product Hunt for a CLI/infra project. |
| Uneed (`uneed.best`), Microlaunch (`microlaunch.net`), Peerlist Launchpad | Small launch platforms, mostly free tiers | Low effort, low risk, some SEO value. Do them in one sitting. |
| SaaSHub (`saashub.com`) | Submit, then verify ownership via the GitHub repo or a DNS/file check. | Also lists alternatives; add the competitor set. |
| StackShare, Slant | Add the tool to the relevant comparison pages | Low traffic now, but pages are old and rank. |
---

## 5. Order of operations

1. **Day 1-2:** verify every repo in section 0 still exists and read each
   `CONTRIBUTING.md`. Skip anything that has moved or died.
2. **Day 2:** open PRs to `punkpeye/awesome-mcp-servers` and
   `awesome-selfhosted/awesome-selfhosted`. These are the two that matter most.
3. **Day 3:** open PRs to `tensorchord/Awesome-LLMOps` and
   `InftyAI/Awesome-LLMOps`.
4. **Day 3:** publish the MCP server to the official MCP Registry, then submit to
   PulseMCP and claim the Glama listing.
5. **Day 5:** submit to AlternativeTo, LibHunt, OpenAlternative, SaaSHub.
6. **Week 2:** `Hannibal046/Awesome-LLM`, `kyrolabs/awesome-langchain`,
   `steven2358/awesome-generative-ai`, `e2b-dev/awesome-ai-agents`.
7. **After the repo has stars:** `ml-tooling/best-of-ml-python` (it has a
   threshold, so submitting early is a wasted PR).
8. **Week 3+:** Product Hunt, DevHunt, Uneed, Microlaunch, Peerlist - all in one
   sitting with the same assets.

---

## 6. Reusable PR body

Keep PRs tiny and factual. Maintainers merge small, obviously-correct PRs.

```markdown
### What

Adds <project> to <section>.

### Why it belongs

<one or two sentences, no adjectives>

- License: Apache-2.0
- Repo: https://github.com/nisaral/DIO
- Last release: v0.4.0
- Docs: <link to the relevant docs file>

### Checklist

- [ ] Entry follows the existing format in <section> exactly
- [ ] Links resolve
- [ ] Placed in the correct position within the section
- [ ] Read CONTRIBUTING.md
```

---

## 7. Tracking table

Duplicate this into your own notes and fill it in as you go.

| Target | PR / submission URL | Opened | Status | Notes |
|---|---|---|---|---|
| punkpeye/awesome-mcp-servers | | | | |
| awesome-selfhosted | | | | |
| tensorchord/Awesome-LLMOps | | | | |
| InftyAI/Awesome-LLMOps | | | | |
| MCP Registry | | | | |
| PulseMCP | | | | |
| Smithery | | | | |
| Hannibal046/Awesome-LLM | | | | |
| kyrolabs/awesome-langchain | | | | |
| steven2358/awesome-generative-ai | | | | |
| ml-tooling/best-of-ml-python | | | | |
| AlternativeTo | | | | |
| LibHunt | | | | |
| OpenAlternative | | | | |
| SaaSHub | | | | |
| Product Hunt | | | | |

---

## 8. Why submissions get rejected

- **Format drift.** One missing backtick, one emoji in the wrong column, and the
  CI fails. Copy an existing line and edit it; never write from memory.
- **Marketing adjectives.** "Blazing fast", "revolutionary", "best-in-class" and
  the like get entries closed on sight, especially on `awesome-selfhosted`.
- **Walls of text.** These lists want one line. The repo has the details; the
  list gets the summary.
- **No source or no license.** Several lists require both inline. Have them
  ready.
- **Being below a threshold.** Some lists have star minimums by design. Check
  before submitting, not after being rejected.
- **Bundling unrelated changes.** One entry, one PR.
- **Pushing after a nudge.** Wait 7 days before following up, once, politely.

---

## 9. Companion channels worth the same effort

Not awesome lists, but the same "durable discovery" category:

- Newsletters: `PyCoder's Weekly`, `console.dev` (dev tools), `MLOps.community`,
  `The Sequence`. Pitch with the dev.to article link, one paragraph.
- Podcasts / Substacks: `Latent Space`, `Interconnects`. Angle: research artifact
  that became a pip-installable product.
- Hugging Face Spaces: a Space that runs `dio demo` against mock backends, so
  "try it" needs no local install.
- Zenodo: the artifact DOI already exists; make sure the Zenodo record links the
  repo and that the README links back, so the two reinforce each other in search.
