"""One worker in a DIO-routed agent swarm.

Each invocation is a single agent session: a byte-stable system prompt (so DIO's
prefix affinity can recognise the session) followed by N turns, all issued
through the gateway. It prints per-turn progress on stderr and one JSON object
on stdout, which makes a swarm's behaviour trivial to aggregate.

Examples::

    # OpenAI-compatible, non-streaming
    python examples/swarm_client.py --url http://127.0.0.1:18077 --agent-id 1 \
        --persona coding --turns 12

    # OpenAI SSE streaming
    python examples/swarm_client.py ... --stream

    # Ollama native /api/chat (ndjson)
    python examples/swarm_client.py ... --api ollama

    # Adversarial burst: long prompts, exercises the SLO admission gate
    python examples/swarm_client.py ... --persona burst --turns 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

# persona -> (stable system prompt, turn prompts cycled through)
PERSONAS: Dict[str, Tuple[str, Sequence[str]]] = {
    "coding": (
        "You are a senior Python engineer working inside a large monorepo. Prefer "
        "small diffs, state the risk of each change, never invent APIs you have not "
        "verified, and say when you are unsure. Keep answers under 120 words.",
        (
            "Refactor the retry path in the gateway proxy so the failure modes are explicit.",
            "Write a regression test for a stream that dies mid-flight.",
            "Why does feeding backend total_tokens into an NLMS learner skew the slope?",
            "Review the last diff for TOCTOU issues and say what you would change.",
            "Add type hints to the scheduler pick() signature and justify each one.",
            "What breaks if a backend returns 200 with an empty choices list?",
        ),
    ),
    "triage": (
        "You are an on-call SRE. Triage fast, rank by blast radius, and always give "
        "one concrete next command. Never speculate without labelling it a guess.",
        (
            "p99 latency doubled ten minutes ago and error budget burn is flat. Go.",
            "One GPU is at 100% KV cache while its peer idles. What do you check first?",
            "A client retried into 503s for a minute. Which header was lying, and why?",
            "The learner's slope for one worker drifted up 3x overnight. Investigate.",
            "Traffic shifted to the slowest worker after a restart. Hypothesise.",
            "Is it the engine, the network, or the router? Give a decision procedure.",
        ),
    ),
    "summarize": (
        "You are a technical writer. Compress without losing the constraint that "
        "matters. Prefer concrete nouns and numbers over adjectives.",
        (
            "Summarise why session affinity matters for KV-cache reuse.",
            "Explain the difference between goodput and throughput in one paragraph.",
            "Describe prefix caching to a new engineer using one analogy.",
            "Summarise what an adaptive router can and cannot fix.",
            "Explain why round-robin is a poor default for heterogeneous GPUs.",
            "Write the release note for a router that learns per-engine latency.",
        ),
    ),
    "burst": (
        "You are a load generator with a job to do. Answer briefly but completely, "
        "and do not refuse the request. Follow the requested format exactly.",
        (
            "Enumerate every failure mode of an HTTP proxy in front of an LLM engine.",
            "List the metrics you would scrape from vLLM every five seconds, and why.",
            "Write the incident timeline template you would use for a GPU fleet.",
            "Describe the load profile of a coding assistant at 9am on a Monday.",
            "Enumerate the ways a KV cache can be wasted, one line each.",
            "List the arguments for and against per-request versus per-session routing.",
        ),
    ),
}


class Rejected(Exception):
    """The gateway answered 4xx/5xx: admission shed, bad request, or upstream error."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail[:200]}")
        self.status = status


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (max(0.0, min(100.0, pct)) / 100.0) * (len(ordered) - 1)
    lo = int(k)
    hi = min(len(ordered) - 1, lo + 1)
    return ordered[lo] * (1.0 - (k - lo)) + ordered[hi] * (k - lo)


async def _openai_json(
    client: httpx.AsyncClient, url: str, history: List[Dict[str, str]], model: str
) -> Tuple[str, Optional[str]]:
    r = await client.post(
        f"{url}/v1/chat/completions",
        json={"model": model, "messages": history, "max_tokens": 48},
    )
    if r.status_code >= 400:
        raise Rejected(r.status_code, r.text)
    data = r.json()
    choices = data.get("choices") or [{}]
    text = (choices[0].get("message") or {}).get("content") or ""
    return text, r.headers.get("X-DIO-Backend")


async def _openai_stream(
    client: httpx.AsyncClient, url: str, history: List[Dict[str, str]], model: str
) -> Tuple[str, Optional[str]]:
    body = {"model": model, "messages": history, "max_tokens": 48, "stream": True}
    chunks: List[str] = []
    backend: Optional[str] = None
    async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
        backend = r.headers.get("X-DIO-Backend")
        if r.status_code >= 400:
            raise Rejected(r.status_code, (await r.aread()).decode("utf-8", "replace"))
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                frame = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(frame, dict) and frame.get("error"):
                raise Rejected(200, json.dumps(frame["error"]))
            delta = ((frame.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if delta:
                chunks.append(delta)
    return "".join(chunks), backend


async def _ollama_chat(
    client: httpx.AsyncClient,
    url: str,
    history: List[Dict[str, str]],
    model: str,
    stream: bool,
) -> Tuple[str, Optional[str]]:
    body = {"model": model, "messages": history, "stream": stream}
    if not stream:
        r = await client.post(f"{url}/api/chat", json=body)
        if r.status_code >= 400:
            raise Rejected(r.status_code, r.text)
        return (r.json().get("message") or {}).get("content") or "", r.headers.get("X-DIO-Backend")

    chunks: List[str] = []
    backend: Optional[str] = None
    async with client.stream("POST", f"{url}/api/chat", json=body) as r:
        backend = r.headers.get("X-DIO-Backend")
        if r.status_code >= 400:
            raise Rejected(r.status_code, (await r.aread()).decode("utf-8", "replace"))
        async for line in r.aiter_lines():
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(frame, dict) and frame.get("error"):
                raise Rejected(200, json.dumps(frame["error"]))
            if isinstance(frame, dict) and frame.get("done"):
                break
            content = (frame.get("message") or {}).get("content")
            if content:
                chunks.append(content)
    return "".join(chunks), backend


async def run_session(args: argparse.Namespace) -> Dict[str, Any]:
    system_prompt, prompts = PERSONAS[args.persona]
    history: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": f"{system_prompt}\n[agent-{args.agent_id} session {args.session_id}]",
        }
    ]
    latencies: List[float] = []
    engines: List[str] = []
    errors = 0
    rejected = 0
    sample = ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        for turn in range(args.turns):
            user = f"{prompts[turn % len(prompts)]} (turn {turn})"
            history.append({"role": "user", "content": user})
            t0 = time.perf_counter()
            try:
                if args.api == "ollama":
                    text, backend = await _ollama_chat(client, args.url, history, args.model, args.stream)
                elif args.stream:
                    text, backend = await _openai_stream(client, args.url, history, args.model)
                else:
                    text, backend = await _openai_json(client, args.url, history, args.model)
            except Rejected as exc:
                if exc.status == 503:
                    rejected += 1
                else:
                    errors += 1
                print(f"  turn {turn}: {exc}", file=sys.stderr, flush=True)
                history.append({"role": "assistant", "content": "(rejected)"})
                continue
            except httpx.HTTPError as exc:
                errors += 1
                print(f"  turn {turn}: transport error: {exc}", file=sys.stderr, flush=True)
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            engines.append(backend or "?")
            sample = text.strip()[:80]
            print(
                f"  turn {turn}: backend={backend} {elapsed_ms:.0f} ms",
                file=sys.stderr,
                flush=True,
            )
            history.append({"role": "assistant", "content": text[:200]})

    switches = sum(1 for a, b in zip(engines, engines[1:]) if a != b)
    return {
        "agent_id": args.agent_id,
        "persona": args.persona,
        "api": args.api,
        "stream": args.stream,
        "turns": args.turns,
        "ok": len(latencies),
        "errors": errors,
        "rejected": rejected,
        "switches": switches,
        "engines": engines,
        "p50_ms": round(_percentile(latencies, 50), 1),
        "p95_ms": round(_percentile(latencies, 95), 1),
        "mean_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "engine_share": {e: engines.count(e) for e in sorted(set(engines))},
        "sample": sample,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="DIO swarm worker (one agent session)")
    ap.add_argument("--url", default="http://127.0.0.1:18077")
    ap.add_argument("--agent-id", type=int, default=1)
    ap.add_argument("--session-id", default="s0")
    ap.add_argument("--persona", choices=sorted(PERSONAS), default="coding")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--model", default="default")
    ap.add_argument("--api", choices=("openai", "ollama"), default="openai")
    ap.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=None,
        help="force streaming (default: off for OpenAI, on for Ollama)",
    )
    ap.add_argument("--no-stream", dest="stream", action="store_false", help="force non-streaming")
    args = ap.parse_args()
    if args.stream is None:
        # Match the real clients: OpenAI requests are non-streaming by default and
        # Ollama's own API streams unless told otherwise.
        args.stream = args.api == "ollama"

    summary = asyncio.run(run_session(args))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
