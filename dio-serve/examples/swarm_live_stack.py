"""Long-running DIO stack for a live agent swarm.

Starts three mock engines (a fast one, a mid one, a slow one), throttles the fast
engine part-way through, and puts a real DIO gateway in front of them on
``--port``. It then stays up so *external* clients -- a swarm of agents, a load
generator, an OpenAI SDK script -- can drive it over HTTP.

Use ``examples/swarm_client.py`` as the worker. Watch the stack's own view of
routing at ``http://127.0.0.1:<port>/debug/metrics`` and, at the end, read the
``--dump`` JSON for the learned model, affinity and admission counters.

Everything is local and offline: the engines are behaviour mocks, so the numbers
describe the *router*, not GPU hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dio.backends import Backend, MockBackendServer
from dio.gateway import DIOGateway

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_swarm_demo import ENGINES, THROTTLE_EVENTS


async def main() -> int:
    ap = argparse.ArgumentParser(description="Live DIO swarm stack (offline, no GPU)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18077)
    ap.add_argument("--base-port", type=int, default=19801, help="first mock engine port")
    ap.add_argument("--duration", type=float, default=600.0, help="seconds to stay up")
    ap.add_argument("--throttle-after", type=float, default=60.0)
    ap.add_argument("--slo-ms", type=float, default=5000.0)
    ap.add_argument("--no-admission", action="store_true", help="disable the SLO gate")
    ap.add_argument("--dump", default="swarm_run.json")
    ap.add_argument("--telemetry-every", type=float, default=20.0)
    args = ap.parse_args()

    servers: Dict[str, MockBackendServer] = {}
    for idx, (name, mult, decode) in enumerate(ENGINES):
        srv = MockBackendServer(
            port=args.base_port + idx, latency_mult=mult, decode_ms_per_token=decode, name=name
        )
        await srv.start()
        servers[name] = srv

    gw = DIOGateway(
        backends=[
            Backend(id=name, base_url=srv.base_url, tier="small", total_vram_mb=24000)
            for name, srv in servers.items()
        ],
        strategy="nlms",
        nlms_mode="dual",
        host=args.host,
        slo_ms=args.slo_ms,
        admission_off=args.no_admission,
        cache_bonus_ms=400.0,
        engine_metrics=False,
        initial_slope=1.0,
        initial_intercept=120.0,
    )

    throttled: List[Tuple[float, str, float]] = []

    async def throttle_after(delay_s: float, name: str, mult: float) -> None:
        await asyncio.sleep(delay_s)
        servers[name].latency_mult = mult
        throttled.append((time.perf_counter(), name, mult))
        print(f"[throttle] {name} degraded to latency_mult={mult}", flush=True)

    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(gw.app, host=args.host, port=args.port, log_level="warning")
    )
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1.0)

    throttle_task = asyncio.create_task(
        throttle_after(args.throttle_after, *THROTTLE_EVENTS[0])
    )

    url = f"http://{args.host}:{args.port}"
    print(
        f"SWARM_READY url={url} engines={','.join(servers)} "
        f"throttle={THROTTLE_EVENTS[0][0]}@{args.throttle_after:.0f}s duration={args.duration:.0f}s",
        flush=True,
    )

    t_end = time.time() + args.duration
    next_telemetry = time.time() + args.telemetry_every
    try:
        while time.time() < t_end:
            await asyncio.sleep(1.0)
            if time.time() >= next_telemetry:
                next_telemetry = time.time() + args.telemetry_every
                m = gw.scheduler.metrics()
                adm = m["admission"]
                aff = m["affinity"]
                slopes = {k: round(v["fast_slope"], 3) for k, v in m["workers"].items()}
                print(
                    f"[telemetry] admitted={adm['admitted']} rejected_slo={adm['rejected_slo']} "
                    f"goodput={adm['goodput_fraction']:.2f} affinity={aff['hit_rate']:.2f} "
                    f"slopes={slopes}",
                    flush=True,
                )
    except asyncio.CancelledError:
        pass

    throttle_task.cancel()
    server.should_exit = True
    await serve_task

    metrics = gw.scheduler.metrics()
    share: Dict[str, int] = {}
    for dec in metrics.get("decisions", []):
        wid = dec.get("worker_id") or dec.get("backend") or "?"
        share[wid] = share.get(wid, 0) + 1
    dump: Dict[str, Any] = {
        "url": url,
        "throttled": [{"at": t, "engine": n, "latency_mult": m} for t, n, m in throttled],
        "engine_share_from_decision_log": share,
        "metrics": metrics,
    }
    Path(args.dump).write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(f"SWARM_DONE dump={args.dump}", flush=True)

    for srv in servers.values():
        await srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
