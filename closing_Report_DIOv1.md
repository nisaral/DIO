DIO v1: Benchmark Report & Scheduler Analysis
Date: January 21, 2026
System: Distributed Inference Orchestrator (v1)
Scheduler: Round-Robin (Naive)
Dataset: ShareGPT (Real-world traces) vs. Synthetic

Executive Summary
DIO v1 has demonstrated exceptional stability and resilience, successfully handling 4,300+ requests with a 0.00% error rate while autoscaling from 1 to 5+ workers under heavy load (200 concurrent users).
However, the system exhibits severe tail latency degradation when switching from uniform synthetic data to real-world inputs (ShareGPT). The 99th percentile latency (p99) degraded by 42%, peaking at 3,000ms, despite the median latency remaining stable at ~860ms.

Conclusion: The Round-Robin scheduling algorithm is mathematically unsuitable for heterogeneous LLM workloads due to Head-of-Line (HoL) Blocking, validating the need for the proposed "Predictive Shortest-Job-First (SJF)" scheduler in v2.


Methodology
Infrastructure:

DIO Manager (Go)
Python Workers (Docker)
Autoscaler (Docker SDK)

Load Generator: Locust (Headless)
Datasets:

Synthetic: Uniform length distribution (Low Variance)
ShareGPT: Real-world conversation traces (High Variance / Power Law)

Metric of Success: Stability (Error Rate) and Tail Latency (p99)

Results: Synthetic vs. Real Data
The following table compares system performance under identical load conditions (200 Users, 5 spawn/sec).
MetricSynthetic DataShareGPT (Real Data)DegradationTotal Requests4,7394,314N/AError Rate0.00%0.00% PASS (Stable)Median Latency (p50)840ms860ms+2.3% (Negligible)Tail Latency (p95)1,800ms2,000ms+11.1%Tail Latency (p99)2,100ms3,000ms +42.8% (CRITICAL)
Visualization: The "Tail Explosion"
Latency (ms)
3000 |                                     ● [p99 SHAREGPT]
2500 |                                      |
2100 |                  ● [p99 SYNTHETIC]   |
1500 |                   |                  |
1000 | ═══════════════════════════════════  [p50 Both]
 500 |
   0 +────────────────────────────────────────────────
       Low Load        Medium Load       High Load

Analysis: Why Round-Robin Failed
Despite the system scaling correctly (new containers spawned successfully), the user experience degraded. This is a queueing theory failure, not a code failure.
The "Head-of-Line" Blocking Problem
In a Round-Robin system, requests are distributed cyclically: Worker A → Worker B → Worker C.
Synthetic Data Scenario:

All requests are roughly the same size
Worker A finishes at the same time as Worker B
Queues drain evenly

ShareGPT Data Scenario:

One request might be 10 tokens ("Hi")
The next might be 2,000 tokens ("Summarize this PDF")

The Failure Scenario:

Worker A receives a Giant Prompt (2s execution)
Worker A immediately receives a Tiny Prompt (50ms execution) because it is "next in line"
The Tiny Prompt sits in the queue for 2 seconds waiting for the Giant Prompt to finish
Result: A 50ms task takes 2,050ms — this drives the p99 metric up, even if the hardware is powerful


Technical Note: Little's Law (L = λW) implies that increasing the Wait Time (W) due to blocking unnecessarily inflates the Queue Length (L), causing memory pressure and latency spikes.


System Resilience (The Good News)
It is important to note that the architecture is sound.
Autoscaling: The logs confirm the Go Manager detected queue depth and spawned 3+ new workers via the Docker Socket during the test.
Networking: New workers successfully registered via gRPC mid-test.
Throughput: The system sustained ~70 RPS without dropping a single packet.

The bottleneck is purely decision making (scheduling), not throughput or stability.


Proposed Mitigation: DIO v2
To address the p99 degradation, v2 will implement Predictive Shortest-Job-First (SJF) scheduling.
Implementation Strategy

Cost Estimation: A Linear Regression model (Cost = 2.1 × Tokens + 50) will predict the duration of incoming requests
Priority Queueing: Instead of a FIFO queue, requests will be sorted by Predicted Cost
Impact: Short tasks will "jump the line," executing immediately on available workers. This mathematically minimizes the Average Waiting Time.

Target Result for v2
Reduce ShareGPT p99 from 3,000ms to <1,500ms