"""Programmatic DIO wrap of any OpenAI-compatible backends."""

from dio import Backend, DIOGateway

# Point these at your vLLM / SGLang / TGI / Ollama servers
gw = DIOGateway(
    backends=[
        Backend(id="gpu0", base_url="http://127.0.0.1:8000", tier="small"),
        Backend(id="gpu1", base_url="http://127.0.0.1:8001", tier="large", total_vram_mb=40000),
    ],
    strategy="nlms",
    nlms_mode="dual",
    slo_ms=30_000,
    admission_off=False,
    port=8085,
)

if __name__ == "__main__":
    # Clients use: OpenAI(base_url="http://localhost:8085/v1", api_key="x")
    gw.run()
