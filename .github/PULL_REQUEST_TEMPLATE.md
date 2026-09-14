# Pull Request

## Summary

<!-- What does this change, and why? One or two sentences. -->

Fixes #<!-- issue number, or delete this line -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation only
- [ ] Refactor / internal cleanup
- [ ] CI / build / tooling

## Which part of the project

- [ ] `dio-serve/` Python product (the shipping package - all new features go here)
- [ ] `DIO/` Go control plane (frozen - bug fixes only, explain why it cannot be done in `dio-serve/`)
- [ ] Docs, config examples, or release metadata only

## How was this tested?

<!-- Paste the exact commands and the result. Do not just check the box. -->

```bash
python -m pytest tests/ -q
ruff check .
```

<!-- If this touches routing or scheduling, include /debug/metrics or /debug/predictions evidence. -->

## Checklist

- [ ] `python -m pytest tests/ -q` passes (35 tests, offline, no GPU needed).
- [ ] New or changed behaviour is covered by a test.
- [ ] `ruff check .` is clean, and `ruff format` was run on the files I touched.
- [ ] Docs updated where relevant (`README.md`, `docs/PRODUCTION.md`, `docs/API.md`, `CHANGELOG.md`).
- [ ] No new `/debug/*` route or debug capability is reachable without the same caveat that all `/debug/*` routes are unauthenticated admin surface.
- [ ] No `/debug/*` path, internal host name, or API key is exposed in examples, screenshots, or logs in this PR.
- [ ] No credentials, private endpoints, or secrets are committed.
- [ ] Commit messages follow Conventional Commits (`feat(scope): ...`, `fix(scope): ...`).
- [ ] Commits are signed off (`git commit -s`) certifying the Developer Certificate of Origin.

## Notes for reviewers

<!-- Anything you are unsure about, follow-ups you deliberately left out, or tradeoffs you made. -->
