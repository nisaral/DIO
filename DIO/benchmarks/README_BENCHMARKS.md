# DIO Benchmark Suite (v1)

This suite validates DIO against standard distributed ML system metrics (vLLM, Ray Serve, etc.).

## Prerequisites

1.  **Install Locust**:
    ```bash
    pip install locust
    ```
2.  **Start DIO**:
    ```bash
    docker-compose up --build
    ```
3.  **(Optional) Download ShareGPT Dataset**:
    If you have `huggingface-cli` installed:
    ```bash
    huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset --local-dir .
    ```
    *If skipped, synthetic data will be used.*

## Running the Benchmarks

Run the full suite:
```bash
python benchmarks/run_v1.py
```

Or run specific tests:

*   **Throughput Baseline**: `python benchmarks/run_v1.py throughput`
*   **Tail Latency**: `python benchmarks/run_v1.py latency`
*   **Autoscaling Stress**: `python benchmarks/run_v1.py autoscale`
*   **Straggler Simulation**: `python benchmarks/run_v1.py straggler`

## Interpreting Results

*   **RPS (Requests Per Second)**: Higher is better.
*   **p99 Latency**: Lower is better. Should stay under 500ms for standard tests.
*   **Failures**: Should be 0. If high during autoscaling, the manager might be dropping requests before new workers are ready.