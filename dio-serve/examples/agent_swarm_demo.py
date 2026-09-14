"""Headline DIO demo: agent swarm against heterogeneous, degrading engines.

Scenario (all local, zero GPU, zero network):

  * three mock engines with different speeds, modelled on a realistic mixed
    fleet (a fast GPU, a slower GPU, and a small/contended one);
  * part-way through the run the FAST engine is throttled (the classic
    co-tenant / thermal-drift event) and stays slow;
  * a swarm of concurrent agents each hold their own multi-turn session.

The same workload is replayed twice: once with plain round-robin over the three
engines, once through DIO. DIO should (a) shift traffic away from the throttled
engine without any configuration change, and (b) keep each session pinned to one
engine so its KV/prefix cache stays warm.

Everything here runs against MockBackendServer, so these numbers demonstrate
*behaviour*, not a performance claim about real hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Dict, List, Sequence, Tuple

from dio.backends import Backend, MockBackendServer
from dio.gateway import DIOGateway

# --------------------------------------------------------------------------- #
# Scenario definition
# --------------------------------------------------------------------------- #

ENGINES: Sequence[Tuple[str, float, float]] = (
    # (name, latency_mult, decode_ms_per_token)
    ("gpu-a", 1.0, 6.0),    # fast GPU
    ("gpu-b", 2.2, 14.0),   # older GPU
    ("gpu-c", 3.6, 20.0),   # small / contended
)

# Applied part-way through the run: the fast GPU gets throttled and stays slow.
# The degradation has to exceed the affinity cache bonus, otherwise staying put is
# the *correct* answer (a warm KV cache is worth ~400 ms in this scenario) and the
# demo shows nothing. 8x is a plausible thermal-throttle / noisy-neighbour event.
THROTTLE_EVENTS = (
    ("gpu-a", 8.0),  # from fastest to slower than every other engine
)

TURNS_PER_AGENT = 16

# The throttling event fires once this many requests have completed, rather than
# after a fixed delay: a wall-clock trigger lands at a different point in each
# replay (the routers run at different speeds), which makes the before/after
# comparison meaningless. Counting requests gives every router the same split.
THROTTLE_AFTER_REQUESTS = 40


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (max(0.0, min(100.0, pct)) / 100.0) * (len(ordered) - 1)
    lo, hi = int(k // 1), min(len(ordered) - 1, int(k // 1) + 1)
    frac = k - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summary(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "n": len(latencies),
        "p50": _percentile(latencies, 50),
        "p95": _percentile(latencies, 95),
        "p99": _percentile(latencies, 99),
        "mean": statistics.fmean(latencies),
    }


# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #

async def _agent_session(
    client,
    resolve_target,
    agent_id: int,
    turns: int,
    results: Dict[str, object],
    progress=None,
) -> None:
    """One multi-turn agent session.

    ``resolve_target(agent_id)`` is called *per turn*, so a per-request load
    balancer behaves like a real one (Nginx's default upstream policy), while a
    stickier policy can pin the whole session to one engine.
    """
    from dio.gateway import _extract_prompt  # reuse DIO's prompt extraction

    session_prefix = (
        f"You are agent-{agent_id}, an autonomous coding assistant working inside a "
        "monorepo. You must be concise, prefer small diffs, always explain the risk of "
        "each change, and never invent APIs that you have not verified in the source. "
        "When you are unsure, say so explicitly and state what you would need to check. "
        f"This is stable session {agent_id} and the conversation below continues it."
    )
    history: List[Dict[str, str]] = [{"role": "system", "content": session_prefix}]
    engines_used: List[str] = []

    for turn in range(turns):
        history.append(
            {"role": "user", "content": f"Turn {turn}: refactor module_{turn} and explain the diff."}
        )
        # Keep the leading system turn in every request so the session prefix stays
        # byte-stable, which is what a real client (and a prefix cache) relies on.
        body = {
            "model": "default",
            "messages": [history[0], *history[-5:]],
            "max_tokens": 32,
        }
        target_url, target_label = resolve_target(agent_id)
        t0 = time.perf_counter()
        resp = await client.post(f"{target_url}/v1/chat/completions", json=body)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code == 200:
            results["latencies"].append(elapsed_ms)
            # DIO reports the pick; when routing directly we already know the target.
            engine = resp.headers.get("X-DIO-Backend") or target_label
            engines_used.append(engine or "?")
            results["timeline"].append((t0, engine or "?", elapsed_ms))
            results["completed"] += 1
            if progress is not None:
                progress(results["completed"])
            history.append({"role": "assistant", "content": "ok"})
        elif resp.status_code == 503:
            results["rejected"] += 1
            history.append({"role": "assistant", "content": "rejected"})
        else:
            results["failed"] += 1

    results["engines_used"].append(engines_used)
    results["prompts"].append(_extract_prompt(history[0]))


async def _run_workload(
    client,
    resolve_target,
    agents: int,
    label: str,
    progress=None,
) -> Dict[str, object]:
    results: Dict[str, object] = {
        "latencies": [],
        "engines_used": [],
        "prompts": [],
        # (perf_counter_at_send, engine, latency_ms) -- lets the report split the
        # run at the throttle event instead of averaging across it.
        "timeline": [],
        "completed": 0,
        "rejected": 0,
        "failed": 0,
    }
    async def one(agent_id: int) -> None:
        await _agent_session(
            client, resolve_target, agent_id, TURNS_PER_AGENT, results, progress
        )

    await asyncio.gather(*(one(i) for i in range(agents)))
    results["label"] = label
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

async def main() -> int:
    parser = argparse.ArgumentParser(description="DIO agent-swarm demo (offline, no GPU)")
    parser.add_argument("--agents", type=int, default=6, help="concurrent agent sessions")
    parser.add_argument("--gateway-port", type=int, default=18099)
    parser.add_argument("--slo-ms", type=float, default=0.0,
                        help="if >0, enable DIO admission control at this budget")
    args = parser.parse_args()

    import httpx

    servers: Dict[str, MockBackendServer] = {}
    base_port = 19701
    for idx, (name, mult, decode) in enumerate(ENGINES):
        srv = MockBackendServer(port=base_port + idx, latency_mult=mult,
                                decode_ms_per_token=decode, name=name)
        await srv.start()
        servers[name] = srv

    gw = DIOGateway(
        backends=[
            Backend(id=name, base_url=srv.base_url, tier="small", total_vram_mb=24000)
            for name, srv in servers.items()
        ],
        strategy="nlms",
        nlms_mode="dual",
        slo_ms=args.slo_ms if args.slo_ms > 0 else 30_000,
        admission_off=args.slo_ms <= 0,
        host="127.0.0.1",  # loopback bind: don't warn about an exposed admin plane
        cache_bonus_ms=400.0,  # make session affinity clearly visible
        engine_metrics=False,  # mocks expose no Prometheus endpoint
        initial_slope=1.0,
        initial_intercept=120.0,
    )

    try:
        import uvicorn

        server = uvicorn.Server(
            uvicorn.Config(gw.app, host="127.0.0.1", port=args.gateway_port, log_level="warning")
        )
        serve_task = asyncio.create_task(server.serve())
        await asyncio.sleep(1.0)
        gw_url = f"http://127.0.0.1:{args.gateway_port}"

        engines = list(servers.items())
        engine_urls = [srv.base_url for _, srv in engines]
        engine_names = [name for name, _ in engines]
        counter = {"i": 0}
        sticky_state: Dict[int, Tuple[str, str]] = {}

        def rr_target(_agent_id: int) -> tuple:
            # Per-request round robin: exactly what an Nginx default upstream does.
            # A session therefore migrates whenever another agent's request
            # interleaves with it, and its prefix cache keeps getting rebuilt.
            idx = counter["i"] % len(engine_urls)
            counter["i"] += 1
            return engine_urls[idx], engine_names[idx]

        def sticky_target(agent_id: int) -> tuple:
            # ip_hash-style pinning: a session stays on one engine for its whole
            # life. Sticky, but blind -- whichever engine it landed on, it keeps.
            if agent_id not in sticky_state:
                idx = agent_id % len(engine_urls)
                sticky_state[agent_id] = (engine_urls[idx], engine_names[idx])
            return sticky_state[agent_id]

        print(f"\nDIO agent-swarm demo | agents={args.agents} turns={TURNS_PER_AGENT}")
        print(f"engines: {', '.join(engine_names)}  "
              f"(throttled mid-run: {THROTTLE_EVENTS[0][0]})")

        def reset_engines() -> None:
            for name, mult, _decode in ENGINES:
                servers[name].latency_mult = mult
            gw.scheduler.reset_stats()

        def make_progress():
            """Fire the throttle once, and record exactly when it fired."""
            state = {"fired": False, "at": 0.0}

            def on_progress(completed: int) -> None:
                if state["fired"] or completed < THROTTLE_AFTER_REQUESTS:
                    return
                state["fired"] = True
                for name, mult in THROTTLE_EVENTS:
                    servers[name].latency_mult = mult
                state["at"] = time.perf_counter()
                print(
                    f"  [throttle] {', '.join(n for n, _ in THROTTLE_EVENTS)} degraded "
                    f"after request {completed}"
                )

            return state, on_progress

        throttle_at = 0.0

        async with httpx.AsyncClient(timeout=120.0) as client:
            print("\n[1/3] plain round-robin (per request, Nginx default) ...")
            state, progress = make_progress()
            rr = await _run_workload(client, rr_target, args.agents, "round_robin", progress)
            throttle_at = state["at"]
            reset_engines()

            print("[2/3] sticky round-robin (ip_hash-style pinning) ...")
            state, progress = make_progress()
            sticky = await _run_workload(
                client, sticky_target, args.agents, "sticky_rr", progress
            )
            reset_engines()

            print("[3/3] the same workload through DIO ...")
            state, progress = make_progress()
            dio_res = await _run_workload(
                client, lambda _aid: (gw_url, None), args.agents, "dio", progress
            )

        server.should_exit = True
        await serve_task

        # ------------------------------------------------------------------ #
        # Report
        # ------------------------------------------------------------------ #
        rows = (
            ("round-robin", rr),
            ("sticky rr", sticky),
            ("DIO", dio_res),
        )

        print("\n" + "=" * 74)
        print("RESULTS  (mock engines - behaviour demo, not a hardware benchmark)")
        print("=" * 74)
        print(f"{'router':<14}{'n':>5}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'mean ms':>10}")
        for label, res in rows:
            s = _summary(res["latencies"])
            print(f"{label:<14}{s['n']:>5}{s['p50']:>10.0f}{s['p95']:>10.0f}"
                  f"{s['p99']:>10.0f}{s['mean']:>10.0f}")

        rr_s, dio_s = _summary(rr["latencies"]), _summary(dio_res["latencies"])
        if rr_s["mean"] and dio_s["mean"]:
            print(f"\nmean improvement vs round-robin: "
                  f"{(1 - dio_s['mean'] / rr_s['mean']) * 100:+.1f}%  "
                  f"({rr_s['mean']:.0f} ms -> {dio_s['mean']:.0f} ms)")
        if rr_s["p99"] and dio_s["p99"]:
            print(f"p99 improvement vs round-robin: "
                  f"{(1 - dio_s['p99'] / rr_s['p99']) * 100:+.1f}%")
        print("(p95/p99 over the whole run are dominated by whatever was in flight "
              "when the engine degraded,\n so they say little here; the window below "
              "is the honest comparison.)")

        def switches(used: List[List[str]]) -> float:
            vals = []
            for seq in used:
                seq = [x for x in seq if x]
                if len(seq) > 1:
                    vals.append(sum(1 for a, b in zip(seq, seq[1:]) if a != b))
            return statistics.fmean(vals) if vals else 0.0

        def share(used: List[List[str]]) -> Dict[str, int]:
            counts: Dict[str, int] = {name: 0 for name in engine_names}
            for seq in used:
                for engine in seq:
                    if engine in counts:
                        counts[engine] += 1
            return counts

        print("\nsession stickiness (engine switches per session, lower is better):")
        for label, res in rows:
            print(f"  {label:<12}: {switches(res['engines_used']):>4.1f}")

        throttled = THROTTLE_EVENTS[0][0]

        def after(res: Dict[str, object]) -> List[tuple]:
            return [row for row in res["timeline"] if row[0] >= throttle_at]

        print("\nengine share, whole run (throttled engine marked *):")
        header = "".join(f"{k + '*' if k == throttled else k:>11}" for k in engine_names)
        print(f"  {'router':<12}{header}")
        for label, res in rows:
            counts = share(res["engines_used"])
            cells = "".join(f"{counts[k]:>11}" for k in engine_names)
            print(f"  {label:<12}{cells}")

        # The informative window: gpu-a is now the *slowest* engine, so a router
        # that has actually learned something should stop sending it work.
        post = {label: after(res) for label, res in rows}
        n_post = min((len(v) for v in post.values()), default=0)
        throttle_txt = ", ".join(f"{n} {m}x slower" for n, m in THROTTLE_EVENTS)
        if n_post >= 8:
            print(f"\nafter the throttle ({n_post} requests, degraded: {throttle_txt}):")
            print(f"  {'router':<12}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}   share")
            for label, _res in rows:
                rows_post = post[label]
                s_post = _summary([r[2] for r in rows_post])
                counts = {k: 0 for k in engine_names}
                for _t, engine, _ms in rows_post:
                    if engine in counts:
                        counts[engine] += 1
                cells = "  ".join(f"{k}={counts[k]}" for k in engine_names)
                print(
                    f"  {label:<12}{s_post['p50']:>10.0f}{s_post['p95']:>10.0f}"
                    f"{s_post['mean']:>10.0f}   {cells}"
                )
            print("(the mean and the share column are what to read here: a router that")
            print(" has learned which engine degraded moves traffic off it.)")
        else:
            print("\n(too few post-throttle requests to split the window)")
        metrics = gw.scheduler.metrics()
        print("\nDIO learned model (per engine):")
        for name, snap in metrics["workers"].items():
            print(f"  {name:<8} slope={snap['fast_slope']:.3f} "
                  f"intercept={snap['intercept']:.0f} "
                  f"updates={snap['updates']} mape={snap['mape_pct']:.1f}%")
        aff = metrics["affinity"]
        print(f"\nDIO affinity: {aff['hits']}/{aff['decisions']} "
              f"({aff['hit_rate'] * 100:.0f}% of picks reused the session's engine)")
        adm = metrics["admission"]
        print(f"DIO admission: admitted={adm['admitted']} "
              f"rejected_slo={adm['rejected_slo']} goodput={adm['goodput_fraction']:.2f}")
        print(f"\nrequests: round-robin completed={rr['completed']} failed={rr['failed']}"
              f" | sticky rr completed={sticky['completed']} failed={sticky['failed']}"
              f" | DIO completed={dio_res['completed']} failed={dio_res['failed']}"
              f" rejected={dio_res['rejected']}")
        if args.slo_ms <= 0:
            print("\n(tip: re-run with --slo-ms 1200 to watch DIO's SLO gate shed load)")
        return 0

    finally:
        for srv in servers.values():
            await srv.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
