"""CLI: dio serve | dio demo | dio bench-smoke"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="dio",
    help="DIO - predictive NLMS orchestrator that wraps vLLM / OpenAI / Ollama engines.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _parse_backends(values: List[str], tiers: List[str], vrams: List[float]) -> list:
    from dio.backends import Backend

    backends = []
    for i, raw in enumerate(values):
        # formats: http://host:port  OR  id=http://host:port  OR  id=url;tier=large
        bid = f"b{i}"
        url = raw
        tier = tiers[i] if i < len(tiers) else "small"
        vram = vrams[i] if i < len(vrams) else 24000.0
        if "=" in raw and not raw.startswith("http"):
            # id=url
            bid, url = raw.split("=", 1)
        if ";tier=" in url:
            url, tier = url.split(";tier=", 1)
        backends.append(
            Backend(
                id=bid.strip(),
                base_url=url.strip(),
                tier=tier.strip(),
                total_vram_mb=vram,
                free_vram_mb=vram,
            )
        )
    return backends


@app.command("serve")
def serve(
    backend: List[str] = typer.Option(
        [],
        "--backend",
        "-b",
        help="Backend base URL (repeatable). Formats: URL | id=URL | id=URL;tier=large",
    ),
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to dio.yaml config file (auto-discovered if omitted)",
    ),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8085, "--port", "-p"),
    strategy: str = typer.Option(
        "nlms",
        "--strategy",
        help="nlms|rls|ewma|static|round_robin|least_loaded",
    ),
    nlms_mode: str = typer.Option("dual", "--nlms-mode", help="dual|single"),
    slo_ms: float = typer.Option(5000.0, "--slo-ms", help="Admission threshold (ms)"),
    admission_off: bool = typer.Option(False, "--admission-off", help="Disable SLO admission rejects"),
    admission_mode: str = typer.Option(
        "empirical",
        "--admission-mode",
        help="absolute|empirical|rank_only (ŷ ranking vs observed-percentile gate)",
    ),
    tokenizer: str = typer.Option(
        "",
        "--tokenizer",
        help="HF tokenizer name for token feature (default: heuristic ⌊len/4⌋)",
    ),
    engine_metrics: bool = typer.Option(
        True,
        "--engine-metrics/--no-engine-metrics",
        help="Scrape backend /metrics (vLLM Prometheus) into hybrid cost",
    ),
    cache_bonus_ms: float = typer.Option(
        200.0,
        "--cache-bonus-ms",
        help="Session/prefix affinity bonus (ms) in joint cost",
    ),
    ablation: str = typer.Option("full", "--ablation"),
    tier: List[str] = typer.Option([], "--tier", help="Tier per backend (same order as --backend)"),
    vram: List[float] = typer.Option([], "--vram", help="Total VRAM MB per backend"),
) -> None:
    """
    Start DIO gateway in front of existing OpenAI-compatible engines.

    Example (two vLLM servers)::

        # GPU0
        python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.2-3B-Instruct --port 8000
        # GPU1
        CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model ... --port 8001

        dio serve -b http://127.0.0.1:8000 -b http://127.0.0.1:8001 --port 8085

        # Client
        curl http://127.0.0.1:8085/v1/chat/completions -H 'Content-Type: application/json' \\
          -d '{"model":"default","messages":[{"role":"user","content":"hi"}]}'
    """
    from dio import DIOConfig, DIOGateway
    from dio.config_file import discover_config, load_config_file

    # Priority: --config > auto-discover dio.yaml > CLI --backend flags
    model_map = {}
    if config_file or (not backend and discover_config()):
        try:
            file_backends, file_cfg, model_map = load_config_file(config_file)
            console.print(f"[green]Loaded config from[/green] {config_file or discover_config()}")
            # CLI flags override config file values
            if backend:
                backends = _parse_backends(backend, tier, vram)
            else:
                backends = file_backends
            cfg = file_cfg
            # Override with explicit CLI flags if they differ from defaults
            if host != "0.0.0.0":
                cfg.host = host
            if port != 8085:
                cfg.port = port
        except FileNotFoundError:
            if not backend:
                console.print("[red]No backends specified and no dio.yaml found.[/red]")
                console.print("  Use: dio serve -b http://localhost:8000")
                console.print("  Or:  dio init  (to create a config file)")
                raise typer.Exit(1)
            backends = _parse_backends(backend, tier, vram)
            cfg = DIOConfig(
                host=host, port=port, strategy=strategy, nlms_mode=nlms_mode,
                slo_ms=slo_ms, admission_off=admission_off, admission_mode=admission_mode,
                tokenizer_name=tokenizer or None, use_tokenizer=bool(tokenizer),
                engine_metrics=engine_metrics, cache_bonus_ms=cache_bonus_ms, ablation=ablation,
            )
    else:
        if not backend:
            console.print("[red]No backends specified.[/red]")
            console.print("  Use: dio serve -b http://localhost:8000")
            console.print("  Or:  dio init  (to create a config file)")
            raise typer.Exit(1)
        backends = _parse_backends(backend, tier, vram)
        cfg = DIOConfig(
            host=host, port=port, strategy=strategy, nlms_mode=nlms_mode,
            slo_ms=slo_ms, admission_off=admission_off, admission_mode=admission_mode,
            tokenizer_name=tokenizer or None, use_tokenizer=bool(tokenizer),
            engine_metrics=engine_metrics, cache_bonus_ms=cache_bonus_ms, ablation=ablation,
        )

    table = Table(title="DIO backends")
    table.add_column("id")
    table.add_column("url")
    table.add_column("tier")
    table.add_column("models", overflow="fold")
    for b in backends:
        models = b.labels.get("models", b.model or "-")
        table.add_row(b.id, b.base_url, b.tier, models)
    console.print(table)

    if model_map:
        mt = Table(title="Model -> Backend routing")
        mt.add_column("model")
        mt.add_column("backends")
        for m, bids in model_map.items():
            mt.add_row(m, ", ".join(bids))
        console.print(mt)

    console.print(
        Panel.fit(
            f"[bold]strategy[/bold]={cfg.strategy}  [bold]nlms[/bold]={cfg.nlms_mode}  "
            f"[bold]slo[/bold]={cfg.slo_ms}ms  [bold]listen[/bold]={cfg.host}:{cfg.port}\n"
            f"OpenAI base_url -> [cyan]http://{cfg.host}:{cfg.port}/v1[/cyan]\n"
            f"Ollama base_url -> [cyan]http://{cfg.host}:{cfg.port}/api[/cyan]",
            title="DIO Serve",
        )
    )
    gw = DIOGateway(backends=backends, config=cfg, model_map=model_map)
    gw.run()


@app.command("demo")
def demo(
    port: int = typer.Option(8085, "--port", "-p"),
    duration: float = typer.Option(20.0, "--duration", "-t", help="Seconds of demo traffic"),
) -> None:
    """
    Zero-GPU demo: spins two mock backends (fast+slow) + DIO, sends traffic, prints metrics.
    Perfect for CI and first-time smoke tests on any laptop / Kaggle CPU node.
    """
    from dio import DIOGateway
    from dio.backends import Backend, MockBackendServer

    async def _run() -> None:
        fast = MockBackendServer(port=19001, latency_mult=1.0, decode_ms_per_token=8.0, name="fast")
        slow = MockBackendServer(port=19002, latency_mult=2.8, decode_ms_per_token=22.0, name="slow")
        await fast.start()
        await slow.start()
        gw = DIOGateway(
            backends=[
                Backend(id="fast", base_url=fast.base_url, tier="small", total_vram_mb=20000),
                Backend(id="slow", base_url=slow.base_url, tier="small", total_vram_mb=16000),
            ],
            strategy="nlms",
            nlms_mode="dual",
            slo_ms=30000,
            admission_off=True,
            port=port,
        )
        import uvicorn

        config = uvicorn.Config(gw.app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        serve_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.8)

        import httpx

        console.print(f"[green]DIO demo listening on http://127.0.0.1:{port}[/green]")
        n_ok = 0
        async with httpx.AsyncClient(timeout=60.0) as client:
            t_end = time.time() + duration
            i = 0
            while time.time() < t_end:
                r = await client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={
                        "model": "mock",
                        "messages": [{"role": "user", "content": f"hello {i} " + ("x" * (i % 50))}],
                        "max_tokens": 32,
                    },
                )
                if r.status_code == 200:
                    n_ok += 1
                    backend = r.headers.get("X-DIO-Backend", "?")
                    if i % 5 == 0:
                        console.print(f"  req {i}: backend={backend} e2e={r.headers.get('X-DIO-E2E-Ms')}ms")
                i += 1

        m = gw.scheduler.metrics()
        console.print(Panel.fit(json.dumps(m["admission"], indent=2), title="Admission"))
        console.print(Panel.fit(json.dumps({k: v.get("fast_slope") for k, v in m["workers"].items()}, indent=2),
                                title="Learned slopes"))
        console.print(f"[bold]Completed OK:[/bold] {n_ok}")
        server.should_exit = True
        await serve_task
        await fast.stop()
        await slow.stop()

    asyncio.run(_run())


@app.command("bench-smoke")
def bench_smoke(
    requests: int = typer.Option(40, "--requests", "-n"),
    concurrency: int = typer.Option(4, "--concurrency", "-c"),
) -> None:
    """Compare NLMS vs round_robin on mock backends (quick paper smoke)."""
    from dio import DIOGateway
    from dio.backends import Backend, MockBackendServer

    async def one_strategy(strategy: str) -> dict:
        fast = MockBackendServer(port=19101, latency_mult=1.0, decode_ms_per_token=6.0, name="f")
        slow = MockBackendServer(port=19102, latency_mult=3.0, decode_ms_per_token=20.0, name="s")
        await fast.start()
        await slow.start()
        gw = DIOGateway(
            backends=[
                Backend(id="f", base_url=fast.base_url),
                Backend(id="s", base_url=slow.base_url),
            ],
            strategy=strategy,  # type: ignore
            admission_off=True,
            slo_ms=60000,
            port=18085,
        )
        import uvicorn

        config = uvicorn.Config(gw.app, host="127.0.0.1", port=18085, log_level="error")
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.6)

        import httpx

        lats = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            sem = asyncio.Semaphore(concurrency)

            async def hit(i: int):
                async with sem:
                    t0 = time.perf_counter()
                    r = await client.post(
                        "http://127.0.0.1:18085/v1/chat/completions",
                        json={
                            "model": "m",
                            "messages": [{"role": "user", "content": f"q{i} " + ("word " * (10 + i % 30))}],
                            "max_tokens": 24,
                        },
                    )
                    lats.append((time.perf_counter() - t0) * 1000)
                    return r.status_code

            codes = await asyncio.gather(*[hit(i) for i in range(requests)])
        server.should_exit = True
        await task
        await fast.stop()
        await slow.stop()
        lats.sort()
        p99 = lats[int(0.99 * (len(lats) - 1))] if lats else None
        return {
            "strategy": strategy,
            "ok": sum(1 for c in codes if c == 200),
            "p50_ms": lats[len(lats) // 2] if lats else None,
            "p99_ms": p99,
            "mape": gw.scheduler.metrics()["prediction"]["mape_pct"],
        }

    async def _run():
        rows = []
        for s in ("round_robin", "nlms"):
            console.print(f"Running {s}...")
            rows.append(await one_strategy(s))
        table = Table(title="bench-smoke")
        for col in ("strategy", "ok", "p50_ms", "p99_ms", "mape"):
            table.add_column(col)
        for r in rows:
            table.add_row(
                r["strategy"],
                str(r["ok"]),
                f"{r['p50_ms']:.0f}" if r["p50_ms"] else "-",
                f"{r['p99_ms']:.0f}" if r["p99_ms"] else "-",
                f"{r['mape']:.1f}",
            )
        console.print(table)

    asyncio.run(_run())


@app.command("init")
def init(
    output: str = typer.Option(
        "dio.yaml",
        "--output",
        "-o",
        help="Output file path",
    ),
    detect: bool = typer.Option(
        True,
        "--detect/--no-detect",
        "-d/-n",
        help="Auto-detect running local engines (Ollama on 11434, vLLM on 8000/8001, SGLang on 30000)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
) -> None:
    """
    Generate a starter dio.yaml config file with optional auto-discovery of local engines.
    """
    import asyncio
    import os

    from dio.config_file import (
        detect_local_backends,
        generate_detected_config,
        generate_example_config,
    )

    if os.path.exists(output) and not force:
        console.print(f"[yellow]{output} already exists.[/yellow] Use --force to overwrite.")
        raise typer.Exit(1)

    discovered = []
    if detect:
        console.print("[dim]Scanning localhost for running inference engines...[/dim]")
        try:
            discovered = asyncio.run(detect_local_backends())
        except Exception:
            discovered = []

    if discovered:
        console.print(f"[green]Discovered {len(discovered)} active inference engine(s):[/green]")
        for b in discovered:
            models_str = ", ".join(b.get("models", []))
            console.print(
                f"  [cyan]*[/cyan] {b['id']} ({b['engine']}) at {b['url']} [dim]models: {models_str}[/dim]"
            )
        content = generate_detected_config(discovered)
    else:
        if detect:
            console.print("[dim]No active local engines found; generating template config.[/dim]")
        content = generate_example_config()

    with open(output, "w", encoding="utf-8") as f:
        f.write(content)

    console.print(f"[green][OK] Created {output}[/green]")
    console.print("")
    console.print("Next steps:")
    console.print(f"  1. Review [cyan]{output}[/cyan]")
    console.print("  2. Run [cyan]dio serve[/cyan] (auto-loads dio.yaml)")


@app.command("config")
def config_show(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Config file path",
    ),
) -> None:
    """Show resolved configuration from dio.yaml (or specify a path)."""
    from dio.config_file import discover_config, load_config_file

    try:
        backends, cfg, model_map = load_config_file(config_file)
    except FileNotFoundError:
        console.print("[yellow]No dio.yaml found.[/yellow] Run [cyan]dio init[/cyan] to create one.")
        raise typer.Exit(1)

    found = config_file or discover_config()
    console.print(f"Config: [cyan]{found}[/cyan]\n")

    table = Table(title="Backends")
    table.add_column("id")
    table.add_column("url")
    table.add_column("tier")
    table.add_column("engine")
    table.add_column("models", overflow="fold")
    for b in backends:
        table.add_row(
            b.id, b.base_url, b.tier,
            b.labels.get("engine", "openai"),
            b.labels.get("models", b.model or "-"),
        )
    console.print(table)

    if model_map:
        console.print("")
        mt = Table(title="Model Routing")
        mt.add_column("model")
        mt.add_column("backends")
        for m, bids in model_map.items():
            mt.add_row(m, ", ".join(bids))
        console.print(mt)

    console.print("")
    console.print(Panel.fit(
        f"strategy={cfg.strategy}  nlms_mode={cfg.nlms_mode}\n"
        f"slo_ms={cfg.slo_ms}  admission={cfg.admission_mode}\n"
        f"engine_metrics={cfg.engine_metrics}  port={cfg.port}",
        title="Scheduler",
    ))


@app.command("version")
def version() -> None:
    from dio import __version__

    console.print(f"dio-serve {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
