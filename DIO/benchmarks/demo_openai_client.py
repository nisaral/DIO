#!/usr/bin/env python3
"""
Artifact demo: OpenAI-compatible client against DIO manager.

  export DIO_BASE_URL=http://127.0.0.1:8085
  python benchmarks/demo_openai_client.py

Uses only the standard library + urllib (no openai package required).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("DIO_BASE_URL", "http://127.0.0.1:8085").rstrip("/")


def chat(prompt: str, model: str = "dio-default", tier: str = "small") -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-DIO-Tier": tier,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    print(f"DIO OpenAI demo → {BASE}")
    try:
        r = urllib.request.urlopen(f"{BASE}/healthz", timeout=5)
        print("healthz:", r.read().decode())
    except Exception as e:
        print("Manager not reachable:", e, file=sys.stderr)
        return 1

    try:
        out = chat("Say hello in one short sentence.")
        print(json.dumps(out, indent=2)[:2000])
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:500], file=sys.stderr)
        return 1
    print("OK — point any OpenAI SDK base_url at", BASE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
