# Security Policy

DIO is an LLM inference gateway. It sits directly in the request path of your prompts and
responses and it controls which backend serves them, so we take vulnerability reports
seriously and we would rather over-explain the security model than let someone deploy it
under a false assumption.

---

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes       |
| < 0.4   | No        |

Security fixes land on the latest 0.4.x release. Older versions, including the pre-0.4
research releases and the legacy Go control plane in `DIO/`, are unsupported and will not
receive fixes. Please reproduce any report against the current version before filing it.

---

## Reporting a vulnerability

**Do not open a public issue, discussion, or pull request for a suspected
vulnerability.** Public disclosure before a fix is available puts every deployment at risk.

Preferred channel: **GitHub Security Advisories** (private, tracked, and gives us a
coordination space).

  https://github.com/nisaral/DIO/security/advisories/new

Fallback if you cannot use GitHub advisories or you need to reach a human directly:

  **nisarkeyush3@gmail.com**

Please include, as far as you can:

- The DIO version (`dio version`) and the Python version.
- Your deployment shape: how DIO is reached, whether a reverse proxy or auth layer is in
  front of it, and whether `/debug/*` is reachable from an untrusted network.
- Which engines back it (vLLM, Ollama, SGLang, TGI, other) and their versions.
- A minimal reproduction: the config file (redact host names and API keys) and the exact
  requests.
- Impact: what an attacker gains, and what they need in order to reach it.
- Any suggested fix, if you have one.

---

## What to expect

| Stage | Target |
|-------|--------|
| Acknowledgement of your report | within 72 hours |
| Initial assessment and severity triage | within 7 days |
| Fix or mitigation plan communicated to you | within 30 days |
| Coordinated public disclosure | 90 days after the report, or sooner by agreement |

We will keep you informed if a timeline slips, and we will credit you in the advisory
unless you ask us not to. This is a small, volunteer-maintained project; these are targets
we hold ourselves to, not contractual guarantees.

---

## Security model and known limitations

This section documents how DIO actually behaves today. Read it before deploying, and read
it before filing a report, because several of these are design decisions rather than bugs.

### DIO performs no authentication

DIO ships with **no authentication of any kind** - no API keys, no bearer tokens, no mTLS,
no per-client identity. Any client that can open a TCP connection to the gateway can use it
and can drain your GPUs. The `api_key` field on a `Backend` exists to authenticate DIO *to*
your inference engines; it does not authenticate clients to DIO.

### DIO binds `0.0.0.0` by default

The default host is `0.0.0.0`, both in the config model (`host: str = "0.0.0.0"` in
`src/dio/config.py`) and in the `dio serve --host` CLI default. `dio demo` and
`dio bench-smoke` bind `127.0.0.1` instead, but the production `serve` path is reachable
from every interface unless you change it.

### Every `/debug/*` route is unauthenticated

There is no separate admin listener and no route-level protection. The debug surface is
split into read-only inspection and state mutation, and both are open:

Read-only, leaks operational internals:

- `GET /debug/metrics` - learned NLMS slopes and intercepts, prediction MAPE, decision counters
- `GET /debug/engine` - scraped engine metrics, including KV cache and queue depth
- `GET /debug/admission` - admission rejection stats and SLO tracking
- `GET /debug/affinity` - session-to-backend affinity state
- `GET /debug/predictions` - recent prediction samples
- `GET /debug/workers` - backend topology, tiers, health, and error counts

State-mutating, and the reason this section exists:

- `POST /debug/reset_stats` - wipes the scheduler's learned state. An unauthenticated caller
  can repeatedly reset the NLMS model, forcing DIO back to cold-start behaviour indefinitely.
- `POST /debug/chaos/vram` - sets an arbitrary `free_vram_mb` for any worker. This is a
  scheduling-control primitive: forged VRAM values steer routing decisions.
- `POST /debug/backends` - hot-registers an arbitrary backend by URL. An unauthenticated
  caller can point DIO at a host they control and receive the prompts and responses that
  DIO then routes there.

Together these mean that **an attacker with network access to the gateway can read your
routing internals, degrade your scheduler at will, and redirect inference traffic to
themselves.** Treat the whole gateway, and `/debug/*` in particular, as an admin surface.

### Required deployment posture

DIO is designed to run on a **private network, behind an authenticating reverse proxy** that
terminates TLS and blocks `/debug/*` from untrusted clients entirely. Concretely:

- Bind `--host 127.0.0.1` (or a private interface) unless you have a specific reason not to.
- Put an authenticating proxy (nginx, Envoy, an API gateway, a service mesh with mTLS) in
  front of it, and require authentication there.
- Block `/debug/*` at the proxy for any non-admin client. Do not rely on "nobody knows the
  path" - the paths are documented.
- Never expose the gateway directly to the public internet.

### Other known limitations

- **No rate limiting or quota.** DIO has no per-client limits and no admission control that
  distinguishes clients. Its admission logic sheds load to protect latency SLOs; that is
  goodput optimization, not abuse protection.
- **No multi-tenant isolation.** All clients share the same backends, scheduler state, and
  affinity table. There is no tenant boundary and no request attribution.
- **Plaintext by default.** DIO speaks HTTP. Prompts and responses are unencrypted unless
  your proxy terminates TLS, and even then the DIO-to-engine hop is plain HTTP in the
  default configuration.
- **`api_key` values and `dio.yaml` are secrets.** A config file can carry engine
  credentials. Keep it out of version control and out of issue reports.
- **Debug endpoints echo backend detail.** `/debug/workers` and `/debug/engine` expose
  internal host names, URLs, and topology. Redact before sharing.
- **Prompt and response content is not logged or stored by DIO**, but it does pass through
  the process in memory and it is forwarded verbatim to the selected engine. Your engine's
  own logging and retention policies apply.
- **No supply-chain provenance yet.** Releases are not currently signed or attested. Pin
  the exact version you deploy and verify it against the published source.

If your threat model needs authentication, tenant isolation, or an admin boundary that DIO
does not yet provide, that has to come from the layer in front of it.
