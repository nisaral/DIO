## Why there is a 0.4.2

`v0.4.1` was tagged from a working tree that still held one uncommitted change:
the robust-learner code. The tag therefore ships a test that its own code does not
satisfy -- `test_an_uncontended_slowdown_is_believed_at_full_speed` fails with
`assert 20 == 0`, so CI is red on that commit -- and the behaviour described under
"Robust learner" in the v0.4.1 notes is not actually in the tagged code.

`v0.4.2` is that release plus the missing change, verified green. Nothing else in
the scheduler moved, so this is a patch, not a feature release.

## Carried from v0.4.1

The six gaps a dogfooding swarm hit on a live gateway, closed rather than
documented (seeding tests in `tests/test_gap_fixes.py`):

- **Robust learner.** A contended sample is clipped against the low quartile of
  recent latencies, with a tolerance that widens while the streak continues; an
  *uncontended* sample is taken at face value at full step size, because the
  scheduler knows how many requests were in flight when the sample was admitted.
  That in-flight depth is the signal that separates queueing from a genuinely
  slow engine. A contended burst used to blow the slope from 2.6 to 168 and
  predict 14.8 s for a 0.25 s request; it now stays at 2.6 / 0.44 s, while a real
  throttle is still believed immediately. `mae`/`mape` keep counting the raw
  error; `clipped_updates` and `contended_updates` report when the guard fires.
- `admission.mode: strict` sheds on the observed tail even when one backend is
  still the clear best option (#23).
- `/debug/affinity` reports `evictions` / `cache_size` / `capacity` (#21).
- One model table behind `/v1/models`, `/api/tags` and routing, so advertised ==
  routable, and a stable Ollama digest across restarts (#20).
- `body_size_cap_bytes` (8 MiB) and `prompt_chars_cap` (1M) return 413 in each
  dialect's envelope (#22).
- `security.data_plane_auth: true` extends `DIO_API_KEY` to `/v1/*` and `/api/*`;
  health probes stay open. Off by default.

## Also in this release

- An architecture diagram (`dio-serve/docs/assets/architecture.svg`) now heads
  both READMEs: the gateway, its four internal pieces, and the stock engines
  behind it.
- Badges corrected to version 0.4.2 and 113 tests.
- Launch kit refreshed: the announcement drafts no longer claim "no auth" (0.4.1
  added the opt-in key), and the awesome-list table records what was submitted,
  what is blocked, and why.

## Verify it

```bash
pip install -e dio-serve
cd dio-serve && python -m pytest tests -q      # 113 passed
```

## Honest demo number

Offline three-way comparison (`examples/agent_swarm_demo.py --agents 8`), eight
concurrent agents, one engine throttled 8x mid-run, mock engines:

| router | session switches | mean (post-throttle) |
|---|---|---|
| round-robin | 10.9 | 1994 ms |
| sticky | 0.0 | 1453 ms |
| **DIO** | **1.0** | **1328 ms** |

Behaviour demo on mock engines, not a hardware benchmark.
