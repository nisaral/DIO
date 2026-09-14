# Launch kit index

Everything needed to take `nisaral/DIO` public, in the order it should be used.
These files are documentation and tooling only - they do not change the product
code, the READMEs, `pyproject.toml`, or `.github/`.

| File | What it is |
|---|---|
| `LAUNCH_PLAYBOOK.md` | The sequenced, dated plan: pre-flight gates, Day 0/1, Week 1/2, Month 1, per-channel hooks, and the anti-patterns that kill a launch. |
| `ANNOUNCEMENTS.md` | Copy-paste-ready drafts: Show HN, r/LocalLLaMA, r/selfhosted, an X post (276/280 chars, verified), LinkedIn, and a dev.to skeleton - plus a fact sheet for replying in comments. |
| `REPO_METADATA.md` | Exact GitHub settings: 150-char description, the 20 topics in priority order, DOI homepage, social-preview spec, and the README changes that convert visitors into stars. |
| `AWESOME_LISTS.md` | Vetted submission targets for this niche, with the section, entry format, contributing rules and real mechanics for each - plus MCP registries and general directories. |
| `gh_repo_setup.ps1` | Idempotent PowerShell script that applies description, topics, homepage, Issues and Discussions via the `gh` CLI. Prints every change; `-DryRun` to preview; no token handling. |

---

## Recommended order of operations

1. **Read `LAUNCH_PLAYBOOK.md` section 0.** Do not launch until the pre-flight
   gates pass. The rest of the kit assumes the repo works on a clean machine.
2. **Run `gh_repo_setup.ps1 -DryRun`**, confirm the diff it reports, then run it
   for real. Details and rationale for every value are in `REPO_METADATA.md`.
3. **Apply the manual metadata** the script cannot do: social preview image,
   pinned repo, the `v0.4.2` release, README demo GIF.
4. **Follow the Day 1 / Week 1 schedule** in the playbook. Open a terminal with
   `dio demo` and the test suite before the first post, and stay in the comments
   for six hours per post.
5. **Submit to the two high-value lists first** (`punkpeye/awesome-mcp-servers`,
   `awesome-selfhosted`) using `AWESOME_LISTS.md`, then the rest over two weeks.
6. **Track the scoreboard** in the playbook weekly, and optimize the referrer
   that produces returning visitors rather than the one with the most upvotes.

## Preview the dry run first

```powershell
.\gh_repo_setup.ps1 -DryRun
.\gh_repo_setup.ps1
```

## Known limits of this kit

- No live network access was available when `AWESOME_LISTS.md` was written, so
  every target carries a confidence value and section 0 gives a command to verify
  each repo exists before you spend time on a PR. Treat medium-confidence rows as
  leads.
- Announcement drafts deliberately make no numeric claim except one, which is the
  committed paired-bootstrap reanalysis in
  `dio-serve/results_reanalysis/REPORT.md`. It is the authors' own reanalysis of
  their own runs, and every draft says so. If you add a measurement, cite the
  script that produced it.
