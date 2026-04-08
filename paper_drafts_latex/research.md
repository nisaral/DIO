

You can mix and match with your current text; I’ve preserved all your key numbers and claims.

***

## 1. Introduction 

```latex
\section{Introduction}

The rapid adoption of Large Language Models (LLMs) has transformed them from research prototypes into the computational backbone of modern AI services.
Production deployments now serve millions of users and billions of tokens per day, and are judged primarily on strict Service-Level Objectives (SLOs) for Time-To-First-Token (TTFT) and tail latency rather than average throughput.
However, the computational cost of serving models such as Llama-3-70B or Mixtral-8$\times$7B exclusively on top-tier accelerators (e.g., NVIDIA A100/H100) is prohibitive.

To reduce Total Cost of Ownership (TCO), operators increasingly deploy \emph{heterogeneous} GPU fleets that combine remnant capacity of older devices (e.g., T4, V100), mid-range inference cards (L4, A10), and spot instances across availability zones.
In such environments, the latency and memory characteristics of different workers are not merely scaled versions of one another: differences in memory bandwidth, tensor-core availability, and VRAM capacity mean that a request served in milliseconds on one GPU can take seconds or even fail on another.
This structural heterogeneity makes it difficult to maintain tight SLOs.

Modern inference engines such as vLLM and Text Generation Inference (TGI) deliver impressive single-node performance via mechanisms like PagedAttention, continuous batching, and optimized kernels.
When deployed in a cluster, however, they typically rely on external load balancers using Round-Robin or Least-Connections policies.
These schedulers implicitly assume that (i) workers are interchangeable and (ii) request cost is roughly uniform.
LLM workloads violate both assumptions: a 10-token summarization query and a 4\,000-token code-generation session impose widely different compute and memory demands, and a worker with 4\,GB of free VRAM is fundamentally different from one with 80\,GB even if their connection counts match.

The mismatch between LLM-specific behavior and generic cluster schedulers manifests in two recurring failure modes.
First, \emph{latency anisotropy} leads to head-of-line (HoL) blocking: short, interactive queries are queued behind long-running generations on slow or saturated GPUs, causing p99 TTFT to balloon by multiples of the median.
Second, \emph{memory blindness} causes reliability failures.
Conventional load balancers cannot distinguish between workers with ample KV-cache space and those near eviction thresholds; routing a long-context request to a memory-saturated worker triggers expensive recomputation or Out-Of-Memory (OOM) crashes that disrupt availability.

\paragraph{Our approach.}
We present \textbf{DIO}, a \emph{model-agnostic} Distributed Inference Orchestrator designed for heterogeneous LLM clusters.
DIO decouples predictive scheduling from execution: a centralized Go control plane learns per-worker performance characteristics \emph{online} using a dual-timescale Normalized Least Mean Squares (NLMS) predictor, and routes requests using a cost function that combines predicted latency, queueing delay, tier/capability constraints, and VRAM pressure.
A Roofline-inspired admission controller enforces memory safety by rejecting requests when all workers are predicted to violate VRAM or latency SLOs, providing proactive backpressure instead of reactive retries.

\paragraph{Contributions.}
This paper makes the following contributions:
\begin{enumerate}
    \item We design a dual-timescale NLMS predictor that learns per-worker ms/token slopes and fixed overheads in $O(1)$ time per request, enabling zero-configuration deployment on heterogeneous GPU fleets without offline profiling.
    \item We integrate this predictor into a Roofline-inspired admission controller and tier-aware cost function that explicitly model VRAM headroom and model capability, eliminating OOM-induced failures while avoiding capability mismatches in multi-model deployments.
    \item We implement a Go-based control plane that orchestrates Python LLM workers over gRPC, adding only 14~$\mu$s median scheduling overhead, and show that it scales linearly to tens of workers while remaining CPU-light.
    \item Across 12 stress tests spanning heterogeneous hardware mixtures, real production traces (ShareGPT, arXiv, Azure Code), and overload scenarios, we demonstrate that DIO reduces p99 latency by up to 63\% relative to Round-Robin baselines and observes zero OOM failures.
\end{enumerate}
Together, these results show that lightweight control-theoretic prediction, when coupled with explicit resource modeling, can substantially improve both tail latency and reliability for LLM serving on heterogeneous clusters.
```

***

## 2. Background and Motivation (())

```latex
\section{Background and Motivation}

\subsection{LLM Serving and Resource Constraints}

LLM inference is dominated by two phases: a \emph{prefill} phase that encodes the prompt, and a \emph{decode} phase that autoregressively generates tokens.
Both phases rely on a KV cache that stores intermediate activations; the cache typically dominates VRAM consumption for long-context workloads.
Modern engines such as vLLM, SGLang, and TGI improve single-node throughput through PagedAttention, continuous batching, and kernel fusion, but they do not perform cluster-level orchestration.
When deployed across multiple GPUs, operators must rely on external schedulers or generic cluster frameworks.

In cluster settings, workloads are highly variable.
Token counts per request span several orders of magnitude, and users mix short chat interactions with long-context summarization or code completion.
Furthermore, hardware heterogeneity is the norm rather than the exception: production fleets frequently combine A100-class GPUs with legacy T4/V100 devices and mid-range cards such as L4s.
The resulting variability in ms/token throughput and available VRAM complicates scheduling.

\subsection{Failure Modes of Existing Schedulers}

Existing deployment stacks typically compose LLM engines with simple load-balancing policies such as Round-Robin (RR), First-Come-First-Served (FCFS), or Least-Loaded (LL).
Cluster frameworks like Ray Serve and Kubernetes rely on coarse metrics---CPU utilization, process count, or connection count---that are only weakly correlated with LLM-specific cost.
As a result, heterogeneous clusters exhibit several recurring failure modes:
\begin{enumerate}
    \item \textbf{Head-of-line blocking.}
    Extreme request-length variance causes short interactive requests to queue behind long-running generations.
    In our traces, a 10-token chat turn stuck behind a 4\,000-token summarization request can inflate p99 TTFT from 860\,ms to over 4\,s.
    Generic schedulers cannot prioritize short jobs without accurate execution-time predictions.
    \item \textbf{Hardware heterogeneity.}
    Real clusters mix GPU generations with 2--4$\times$ variance in ms/token (e.g., $0.5$--$1.0$\,ms/token on A100 vs.\ $2.0$--$3.0$\,ms/token on T4).
    Queue length and connection count become misleading across throughput classes: a slow worker with a short queue may still be slower than a fast worker with a longer one.
    \item \textbf{Cold starts and drift.}
    Elastic scaling introduces new workers whose performance is initially unknown, and long-running workers experience performance drift due to thermal throttling and co-located jobs.
    Systems that rely on static profiles or offline benchmarking cannot adapt quickly enough, leading to prolonged periods of suboptimal routing.
    \item \textbf{Memory safety and KV-cache overflow.}
    VRAM-constrained workers routinely operate near 90\% utilization.
    Routing a long-context request to such workers triggers eviction, recomputation, or OOM crashes that propagate as timeouts and retries.
    Existing systems largely react \emph{after} failure rather than enforcing proactive admission control.
    \item \textbf{Load spikes and backpressure.}
    Traffic bursts---for example, from viral events---overwhelm reactive autoscalers.
    Without predictive admission, schedulers attempt to queue unbounded work, causing SLO violations across all requests and risking control-plane collapse.
    \item \textbf{Control-plane scalability.}
    Prior heterogeneous schedulers such as NexusSched rely on matrix-based Recursive Least Squares (RLS) predictors with $O(N^2)$ per-update cost, which limits scalability and increases per-request scheduling latency.
\end{enumerate}

Table~\ref{tab:systems-comparison} summarizes the capabilities of representative systems.
No existing solution simultaneously offers per-request online prediction, explicit VRAM admission control, and low-overhead scheduling in heterogeneous environments.

\begin{table}[t]
\centering
\begin{tabular}{lcccc}
\toprule
System & Hetero & Admission & Online & Overhead \\
\midrule
vLLM / SGLang & No & Reactive & No & N/A \\
Ray Serve     & Coarse & No & No & ms \\
NexusSched    & Yes & No & Offline & $O(N^2)$ \\
CoCoServe / Loong & Partial & Instance & No & Unreported \\
\textbf{DIO}  & Yes & Per-request & Online & 14~$\mu$s \\
\bottomrule
\end{tabular}
\caption{LLM serving systems.
DIO uniquely combines heterogeneity support, proactive VRAM-aware admission, and low-overhead online prediction.}
\label{tab:systems-comparison}
\end{table}

\subsection{Design Goals}

These gaps motivate the design of DIO.
Our orchestrator is guided by five goals:
\begin{enumerate}
    \item \textbf{Zero-configuration adaptation.} Learn per-worker performance online from telemetry, avoiding any offline profiling or engine modifications.
    \item \textbf{Proactive memory safety.} Enforce VRAM-aware admission to prevent KV-cache overflow and OOM cascades rather than reacting after failures.
    \item \textbf{Tier- and capability-aware routing.} Handle multi-model deployments by encoding capability constraints (e.g., long-context vs.\ short-context models) directly in the cost function.
    \item \textbf{Sub-millisecond decisions.} Maintain $O(1)$ per-request computational cost in the control plane so that scheduling overhead remains negligible relative to inference time.
    \item \textbf{Non-invasive integration.} Operate as an external control plane that orchestrates existing LLM engines via gRPC without requiring kernel or runtime changes.
\end{enumerate}
The remainder of the paper describes how DIO realizes these goals and evaluates its effectiveness in heterogeneous clusters.
```

***

## 3. System Overview (architecture + math) – ()

```latex
\section{System Overview}

DIO consists of a centralized control plane implemented in Go and a set of stateless Python workers running vLLM-based inference.
The control plane is responsible for request admission, per-worker latency prediction, and cost-based routing; workers execute model inference and report telemetry.
Figure~\ref{fig:architecture} shows the high-level architecture.

\subsection{Design Evolution}

DIO did not start as a control-theoretic scheduler; the current design emerged after simpler policies failed under realistic heterogeneity and load.

\paragraph{v0: least-connections wrapper.}
The initial prototype wrapped vLLM behind a reverse proxy that routed requests according to the number of outstanding connections per worker.
On a single GPU or a homogeneous group of similar GPUs this behaved acceptably, but in mixed A100/L4/T4 clusters the slowest worker accumulated the same number of in-flight requests as the fastest one.
This inflated p99 TTFT by up to $3.5\times$ due to head-of-line blocking on the weaker device.

\paragraph{v1: static profiling.}
We next benchmarked each worker on a synthetic workload and used the resulting ms/token slopes to weight queue length when routing.
This reduced the worst misroutes but quickly became inaccurate: thermal throttling, background jobs, and driver changes slowed individual GPUs by 30--50\% within minutes, and the scheduler had no mechanism to adapt.
Tail latency spikes reappeared as profiles went stale.

\paragraph{v2: online per-worker latency learning.}
To eliminate offline profiling, DIO moved to an online predictor.
Each completed request reports its token count and latency, and a normalized LMS (NLMS) loop updates a per-worker slope and bias in $O(1)$ time.
Routing decisions are driven by predicted service times rather than static heuristics, allowing DIO to track both short-term interference and longer-term drift.

\paragraph{v3: tier- and memory-aware cost.}
The current design extends v2 by augmenting the cost function with (i) tier labels on requests and workers to encode model capability, (ii) VRAM-aware soft penalties and hard admission gates to avoid routing to memory-saturated workers, and (iii) a lightweight KV-cache affinity bonus based on prompt prefixes.
The learning rule remains unchanged; only the cost function is extended.
The result is a control plane that remains simple enough to run on a single CPU node while being aware of both compute speed and memory headroom for each worker.

\subsection{System Architecture}

DIO has two primary layers: a high-throughput Go control plane and a data plane consisting of Python inference workers.
The control plane makes all routing and admission decisions; workers simply execute model inference and piggy-back telemetry on responses.

\subsubsection{Control Plane (Go)}

The control plane runs as a single process with three responsibilities:

\begin{enumerate}
    \item \textbf{Request API.}
    A lightweight HTTP/gRPC endpoint accepts generation requests containing the prompt bytes, optional tier label, and sampling parameters.
    The control plane approximates the prompt token count from its byte length and forwards the request to the scheduler.

    \item \textbf{Online predictors.}
    For each registered worker $w$, the control plane maintains a small state vector comprising (i) a fast and slow slope estimate in ms/token, (ii) a bias term capturing fixed overheads, (iii) a moving average of observed service time, and (iv) static metadata such as tier and total VRAM.
    These parameters are updated on every completed request using a dual-timescale NLMS rule (\S\ref{sec:nlms-model}).

    \item \textbf{Scheduler and admission controller.}
    Upon arrival of a new request $r$, the scheduler scans all healthy workers, computes a scalar cost for assigning $r$ to each worker that satisfies capability constraints, and selects the worker with minimum cost.
    The cost combines predicted execution time, queueing delay, tier penalties, VRAM pressure penalties, and KV-cache affinity bonuses (\S\ref{sec:scheduling}).
    If all workers exceed VRAM or latency thresholds, the admission controller rejects the request.
\end{enumerate}

Worker metadata is stored in a lightweight BoltDB-backed registry, and a health monitor tracks worker liveness via periodic probes.
The control plane is written in Go to exploit goroutines and a low-latency garbage collector, enabling tens of thousands of concurrent HTTP and gRPC calls with minimal per-request allocation.

\subsubsection{Data Plane (Python Workers)}

Workers are Python processes that each host a single LLM instance (e.g., a 7B model on an A100, a 4-bit 8B model on an L4, or a smaller model on a T4).
Each worker exposes a gRPC service \texttt{InferenceWorker} with two operations: \texttt{Generate} and \texttt{Health}.
The control plane invokes \texttt{Generate} with the prompt payload and sampling parameters; the worker responds with generated tokens and telemetry including request latency (TTFT and end-to-end), token counts, and current free VRAM obtained via NVML.

Internally, a worker contains a thin inference loop around vLLM (or a compatible engine) and performs no cluster-level scheduling.
This keeps the worker implementation simple and allows DIO to orchestrate different engines as long as they satisfy the same gRPC contract.
Telemetry is attached to normal inference responses rather than sent via separate heartbeats, which reduces cross-traffic and keeps the feedback loop tight.

\subsection{Latency Physics: Dual-Timescale NLMS}
\label{sec:nlms-model}

DIO predicts per-worker latency using a lightweight linear model updated by a dual-timescale NLMS filter.
For a request $k$ with $N_k$ tokens served on worker $w$, we model the end-to-end latency $y_k$ as
\[
    \hat{y}_k = s_w N_k + b_w, \tag{1}
\]
where $s_w$ (ms/token) is a processing slope capturing throughput and $b_w$ is a fixed overhead capturing tokenization, KV-cache allocation, and gRPC serialization.

Given the observed latency $y_k$, the prediction error is $e_k = y_k - \hat{y}_k$.
To avoid large updates for very long prompts, we normalize the error by the input magnitude:
\[
    \nabla_{\text{norm}} = \frac{e_k}{N_k + \epsilon}, \tag{2}
\]
where $\epsilon$ is a small constant for numerical stability.

We maintain two slope estimates per worker: a fast estimate $s^{\text{fast}}_w$ that tracks transient interference and a slow estimate $s^{\text{slow}}_w$ that captures longer-term drift.
They are updated as
\[
    s^{\text{fast}}_w \leftarrow s^{\text{fast}}_w + \mu_{\text{fast}} \,\nabla_{\text{norm}}, \tag{3}
\]
\[
    s^{\text{slow}}_w \leftarrow s^{\text{slow}}_w + \mu_{\text{slow}} \,\nabla_{\text{norm}}, \tag{4}
\]
with learning rates $\mu_{\text{fast}} = 0.1$ and $\mu_{\text{slow}} = 0.01$.
The effective slope used for scheduling is a convex combination
\[
    s^{\text{eff}}_w = 0.8\,s^{\text{fast}}_w + 0.2\,s^{\text{slow}}_w, \tag{5}
\]
which acts as a band-pass filter: it is responsive to short-term changes while smoothing out noise.

The bias term is updated more conservatively:
\[
    b_w \leftarrow b_w + \mu_b \,\nabla_{\text{norm}}, \tag{6}
\]
with $\mu_b = 0.005$, reflecting the relative stability of fixed overheads.
Standard NLMS theory ensures bounded-input bounded-output stability for $0 < \mu < 2$, and empirical tuning on A100 and L4 hardware showed that these values strike a balance between convergence speed and stability.
The per-request computational cost of these updates is $O(1)$: a few scalar additions and multiplications per worker independent of cluster size.

\subsection{Tier- and Roofline-Aware Scheduling}
\label{sec:scheduling}

The scheduler uses the NLMS predictions to compute a scalar cost $S(r, w)$ for assigning a request $r$ to worker $w$.
For a request with approximate token count $N$ and tier label $T_r$, we first estimate the execution time using the effective slope:
\[
    \hat{t}_w(N) = s^{\text{eff}}_w N + b_w. \tag{7}
\]

Let $Q_w$ denote the number of queued requests at worker $w$, and let $\bar{t}_w$ be its moving-average service time.
With continuous batching of size $B$, we approximate the expected waiting time as
\[
    \text{wait}_w = \frac{Q_w}{B}\,\bar{t}_w, \tag{8}
\]
a batching-aware variant of Little's Law.

Requests and workers may carry tier labels $T \in \{\text{small}, \text{large}, \ldots\}$.
We encode a tier penalty
\[
\text{tierCost}(r, w) =
\begin{cases}
0, & T_r = T_w, \\
500, & T_r = \text{small},\ T_w = \text{large}, \\
\infty, & T_r = \text{large},\ T_w \neq \text{large},
\end{cases}
\tag{9}
\]
ensuring that large-tier requests are only routed to capable workers, while small-tier requests preferentially use smaller models when possible.

To model VRAM pressure, we track each worker's free VRAM $\text{free}_w$ and total VRAM $\text{total}_w$.
When $\text{free}_w$ falls below a threshold (4\,GB in our prototype), we add a soft penalty
\[
    \text{vramCost}_w =
    \mathbf{1}[\text{free}_w < 4\text{GB}]
    \left(1 - \frac{\text{free}_w}{\text{total}_w}\right)\cdot 1000,
    \tag{10}
\]
discouraging but not forbidding routing to memory-constrained workers.
If a separate Roofline model predicts that the KV cache for request $r$ would exceed available VRAM on worker $w$, we set $\hat{t}_w(N) = \infty$, implementing hard admission rejection.

Finally, DIO exploits KV-cache locality via a lightweight prefix cache that maps the first 100 bytes of prompts to the worker that last served a matching prefix.
If the prefix hash $H_r$ of request $r$ equals the cached entry for worker $w$, we apply a cache affinity bonus
\[
    \text{cacheBonus}(r, w) = 200\ \text{ms}. \tag{11}
\]

The total cost of assigning $r$ to $w$ is then
\[
    S(r, w) = \hat{t}_w(N) + \text{wait}_w + \text{tierCost}(r, w)
              + \text{vramCost}_w - \text{cacheBonus}(r, w). \tag{12}
\]
Mathematical derivation of the **Normalized Least Mean Squares (NLMS)** algorithm for DIO latency prediction focuses on stabilizing the learning process across variable request lengths. Traditional Least Mean Squares (LMS) can suffer from "gradient explosion" when prompts vary from a few tokens to several thousand, as the weight update is directly proportional to the magnitude of the input vector. NLMS addresses this by normalizing the update by the power of the input.

### 1. The Latency Physics Model

DIO models the latency  of a request  on a specific worker  as a linear function of the token count :


* ****: The processing slope (ms/token), representing the marginal cost of adding one token.
* ****: The fixed overhead (intercept), capturing constant factors like kernel launch times and network base latency.

### 2. Derivation of the NLMS Update Rule

The goal of the NLMS filter is to adjust the slope  and bias  at each step  to minimize the squared prediction error: .

**Standard Gradient Descent:**
In simple LMS, the update would be:



Where  and . However, if  is very large (e.g., a 4000-token prompt), the update magnitude becomes too large, causing instability.

**Normalization for Stability:**
To ensure stability, NLMS normalizes the update by the squared norm of the input vector . For DIO's primary slope update, this simplifies to:



Here,  is a small regularization constant to prevent division by zero when token counts are minimal.

### 3. Dual-Timescale Adaptation

To distinguish between transient "jitter" (e.g., temporary network congestion) and permanent "drift" (e.g., GPU thermal throttling or background workload interference), DIO maintains two separate slope estimates:

1. **Fast Scale ():** Reacts quickly to immediate feedback using a high learning rate ().


2. **Slow Scale ():** Tracks long-term hardware performance trends using a low learning rate ().



**Final effective scheduling slope:**
The orchestrator uses a weighted sum of these two scales to drive routing decisions, typically biased toward the fast scale for agility:



This acts as a mathematical **band-pass filter**, allowing the system to ignore singular outliers while adapting rapidly to persistent changes in worker capability.

### 4. Application in Scheduling

This derived slope is then used in the cost-based scheduling function to estimate the total "work" a request represents:



This ensures that requests are routed not just to workers with the fewest connections, but to the worker that will physically complete the specific task the fastest based on its learned performance profile.
Algorithm~\ref{alg:dio-scheduling} shows the resulting scheduling rule.
The scheduler performs a linear scan over the set of healthy workers, evaluating $S(r, w)$ in constant time per worker.
For typical cluster sizes of 4--32 workers, this yields sub-microsecond scheduling latency.

\begin{algorithm}[t]
\caption{Tier- and Roofline-Aware Scheduling (DIO)}
\label{alg:dio-scheduling}
\begin{algorithmic}[1]
\Require Request $r$ with prompt bytes $D$, tier $T_r$
\Require Worker set $W$
\State $N \leftarrow \lfloor |D| / 4 \rfloor$ \Comment{Approximate tokens}
\State $w^\star \leftarrow \text{null}$, $S_\text{min} \leftarrow \infty$
\State $H_r \leftarrow$ prefix hash of first 100 bytes of $D$
\ForAll{workers $w \in W$}
    \If{not $w.\text{healthy}$}
        \State \textbf{continue}
    \EndIf
    \If{$T_r = \text{large}$ and $w.\text{tier} \neq \text{large}$}
        \State \textbf{continue} \Comment{Hard capability constraint}
    \EndIf
    \State $(\hat{t}_w, \bar{t}_w) \leftarrow \textsc{PredictLatency}(w, N)$
    \If{$\hat{t}_w = \infty$}
        \State \textbf{continue} \Comment{Hard VRAM admission block}
    \EndIf
    \If{$\bar{t}_w = 0$}
        \State $\bar{t}_w \leftarrow \hat{t}_w$
    \EndIf
    \State $Q_w \leftarrow w.\text{pending}$, $B \leftarrow 8.0$
    \State $\text{wait} \leftarrow (Q_w / B) \cdot \bar{t}_w$
    \State $\text{tierCost} \leftarrow 0$
    \If{$T_r = \text{small}$ and $w.\text{tier} = \text{large}$}
        \State $\text{tierCost} \leftarrow 500$
    \EndIf
    \State $\text{vramCost} \leftarrow 0$
    \If{$w.\text{freeVRAM} < 4096\ \text{MB}$}
        \State $\text{pressure} \leftarrow 1 - \frac{w.\text{freeVRAM}}{w.\text{totalVRAMMB}}$
        \State $\text{vramCost} \leftarrow \text{pressure} \cdot 1000$
    \EndIf
    \State $\text{cacheBonus} \leftarrow 0$
    \If{$\text{PrefixCache}[H_r] = w.\text{id}$}
        \State $\text{cacheBonus} \leftarrow 200$
    \EndIf
    \State $S_w \leftarrow \hat{t}_w + \text{wait} + \text{tierCost} + \text{vramCost} - \text{cacheBonus}$
    \If{$S_w < S_\text{min}$}
        \State $S_\text{min} \leftarrow S_w$, $w^\star \leftarrow w$
    \EndIf
\EndFor
\State \Return $w^\star$
\end{algorithmic}
\end{algorithm}
```

***

## 4. Experimental Evaluation (())

```latex
\section{Experimental Evaluation}

We evaluate DIO on two hardware configurations and a mix of synthetic and real-world workloads.
Our study addresses four questions:
\textbf{RQ1}~(Scalability): Does the Go-based control plane remain lightweight as the number of workers grows?
\textbf{RQ2}~(Performance): Does NLMS-based predictive scheduling improve tail latency and SLO attainment compared to standard schedulers?
\textbf{RQ3}~(Safety): Can DIO prevent Out-of-Memory (OOM) failures under VRAM pressure?
\textbf{RQ4}~(Adaptability): How effectively does DIO handle hardware heterogeneity, cold starts, and overload?

\subsection{Experimental Methodology}

\subsubsection{Hardware Testbed}

We use two environments:

\paragraph{Edge configuration (L4).}
A Google Cloud \texttt{g2-standard-4} instance with one NVIDIA L4 GPU (24\,GB GDDR6), 4 vCPUs, and 16\,GB RAM.
We emulate heterogeneity by injecting a synthetic 500\,ms prefix processing delay into 50\% of the workers for test T2.

\paragraph{Data-center configuration (A100).}
An \texttt{a2-highgpu-1g} instance with one NVIDIA A100 GPU (80\,GB HBM2e), 12 vCPUs, and 85\,GB RAM.
We run four concurrent model replicas on the A100 to create realistic PCIe contention and VRAM pressure.

All experiments use CUDA 12.3, the latest compatible NVIDIA driver, and vLLM as the underlying engine.

\subsubsection{Workloads and Datasets}

We combine synthetic probes with replayed production traces.

\paragraph{Synthetic tests.}
Microbenchmarks T1--T7 use fixed or log-normal token length distributions to probe NLMS convergence, heterogeneity, control-plane scalability, and Roofline admission.
Each run lasts 120\,s with a closed-loop load generator (Locust) ramping from 30 to 500 concurrent users.

\paragraph{Production traces.}
We replay three real workloads:
\emph{ShareGPT} (T9) is a conversational trace with multi-turn dialogues and variable output lengths;
\emph{arXiv Summarization} (T10) consists of long-context summarization requests with prompts up to 4\,000 tokens;
\emph{Azure Code} (T11) contains bursty code-generation tasks with highly variable decoding lengths.
We preserve the original inter-arrival times and token length distributions.

\paragraph{SLO definition.}
Unless otherwise noted, we consider a request SLO-compliant if its TTFT is at most 2\,s.
We report p50, p95, and p99 latency as well as SLO attainment and goodput (throughput of SLO-compliant requests).

\subsubsection{Baselines and Metrics}

We compare DIO against two common schedulers:

\begin{itemize}
    \item \textbf{Round-Robin (RR).} Default multi-GPU balancer for vLLM, routing requests cyclically without regard to worker state.
    \item \textbf{Least-Loaded (LL).} Heuristic used by Ray Serve, routing to the worker with the fewest active connections.
\end{itemize}

Where applicable, we also normalize results against numbers reported for NexusSched, SGLang, and DistServe.
Our primary metrics are:
(i) p99 TTFT;
(ii) SLO attainment;
(iii) goodput;
(iv) failure rate (timeouts, gRPC errors, VRAM OOM);
and (v) control-plane scheduling overhead measured at the DIO manager.

Each data point is averaged over ten runs; we report mean and standard deviation.

\subsection{Scalability and Control-Plane Overhead (RQ1)}

We stress the control plane using test T7 on the A100 node with up to 32 concurrent workers.
Workers are lightweight stubs that emulate inference latency using configurable ms/token slopes; this isolates control-plane cost from GPU computation.

Figure~\ref{fig:cp-scale} reports per-request scheduling overhead and achieved throughput as the number of workers increases.
DIO maintains a median scheduling overhead of 14~$\mu$s and a p99 of 1.9\,ms at 32 workers, with no observed failures across 1\,664 requests.
Throughput scales linearly to 27.9\,req/s, after which the single A100 saturates.
These results confirm the expected $O(|W|)$ scaling of the linear-scan scheduler and show that control-plane cost is negligible relative to inference time.

The memory footprint of the manager remains modest.
Each worker's state occupies approximately 256\,B (slope, bias, counters, metadata), and the prefix cache is capped at 10\,000 entries ($\approx$80\,KB).
A single manager can therefore handle over 10\,000 workers with under 3\,MB of heap allocation, leaving ample headroom for additional features.

\subsection{End-to-End Performance on Production Traces (RQ2)}

We next evaluate DIO on the A100 configuration with four concurrent workers using the three production traces.
Figure~\ref{fig:trace-results} summarizes p99 latency (top row) and SLO attainment (bottom row).

On \emph{ShareGPT}, DIO achieves a p99 latency of 2.38\,s compared to 5.40\,s under Round-Robin.
By predicting generation length and queueing delay, DIO prevents short chat turns from being queued behind long conversations.
SLO attainment improves from 78.5\% (RR) to 96.2\%.

The differences are more pronounced on \emph{arXiv Summarization}, which stresses VRAM capacity.
RR blindly routes memory-intensive prompts to saturated workers, causing KV-cache thrashing and driving p99 latency to 82.5\,s.
DIO's VRAM-aware routing identifies pressure and steers requests to workers with headroom, maintaining a p99 of 13.5\,s---a 6$\times$ reduction---while achieving 71\% SLO attainment versus 34\% for RR.

On \emph{Azure Code}, decoding times are highly unpredictable, and RR frequently overloads workers, yielding only 12\% SLO attainment.
DIO's adaptive predictor corrects slope estimates in real time, balancing load across workers and raising SLO attainment to 88\%.

Across all three traces, DIO processes a similar number of total requests as RR and LL, but its goodput (SLO-compliant throughput) is 2.1--7.3$\times$ higher.
This distinction is critical in production environments where SLA violations incur penalties or user churn.

\subsection{Handling Hardware Heterogeneity (RQ4)}

To study DIO's behavior in heterogeneous environments, we run test T2 on the L4 configuration with two workers.
One worker is designated ``fast''; the other incurs a 500\,ms artificial prefix delay on each request.
RR distributes requests 50/50, ignoring the performance difference.

Figure~\ref{fig:hetero-routing} shows the resulting routing distribution and p99 latency.
Under RR, queues quickly build up on the slow worker, effectively capping cluster performance at its throughput and increasing p99 latency by 41\%.
DIO's NLMS predictor rapidly detects the discrepancy between the fast and slow slopes.
Within 15--20 requests per worker (about 8\,s in our setup), the predictor converges, and the scheduler routes approximately 75\% of requests to the fast worker.
This adaptive routing reduces aggregate p99 latency by 41\% relative to RR and largely hides the straggler from clients.

\subsection{Ablation Study: Component Contribution (RQ2/RQ3)}

We perform an ablation study on the L4 platform to quantify the contribution of DIO's individual components: VRAM-aware Roofline admission, queue-aware wait time, and tier-aware routing.
We start from the full system and disable one component at a time.

Figure~\ref{fig:ablation} and Table~\ref{tab:ablation} summarize the results on the arXiv workload.
Removing VRAM-aware logic (\texttt{-VRAM}) yields a 23\% failure rate due to OOM errors; routing continues to send long-context requests to saturated workers until KV-cache overflow occurs.
Removing queue-awareness (\texttt{-Queue}) preserves memory safety but causes p99 latency to spike to 89\,s, confirming that connection counts are insufficient proxies for load.
Disabling tier labels (\texttt{-Tiers}) increases p99 latency by roughly 22\% because small requests are occasionally routed to large-tier workers with heavier models.

Only the full configuration achieves both low tail latency (53\,s p99) and zero failures, demonstrating that predictive routing must be coupled with explicit VRAM modeling and queue-aware cost estimation.

\subsection{Resilience and Cloud-Native Behavior (RQ3/RQ4)}

Finally, we assess DIO's behavior under cloud-native scenarios using tests T3 and T4.

In the \emph{Cold Start} scenario (T3), a new L4 worker joins an existing cluster at runtime.
The NLMS predictor initializes its parameters from the first telemetry packet and converges within approximately 1.2\,s.
The new worker quickly takes on its fair share of load without causing a prolonged p99 spike.

In the \emph{Roofline Stress} scenario (T4), we flood the L4 cluster with 200\,req/s, far beyond its physical capacity.
Baseline schedulers attempt to queue all requests, leading to 100\% SLO violation and eventual resource exhaustion.
By contrast, DIO's admission controller proactively rejects excess requests once predicted wait times exceed the SLO.
During overload, the manager returns HTTP~503 responses with a \texttt{Retry-After} header indicating the estimated queue-drain time.
This ``load shedding'' behavior acts as a circuit breaker: accepted traffic continues to meet SLOs, and the system remains responsive even under $3\times$ overload.

Across convergence, heterogeneity, cold start, and overload tests, DIO maintains zero failure rate for admitted requests, demonstrating robustness in cloud-native deployment scenarios.
```

***

## 5. Discussion and Conclusion (())

```latex
\section{Discussion}

DIO demonstrates that a centralized, prediction-based control plane can substantially improve both tail latency and reliability for LLM serving on heterogeneous clusters without modifying underlying engines.
The dual-timescale NLMS predictor adapts quickly to runtime interference and hardware drift, while the Roofline-inspired cost function enforces memory safety and tier constraints.

\paragraph{Limitations.}
The current design assumes a single logical control plane.
Although our experiments show that one manager can handle tens of workers with negligible overhead, deployments with hundreds or thousands of workers may require sharding or a distributed control-plane architecture.
In addition, we assume that each worker hosts a single model and expose only coarse tier labels to the scheduler; multi-model workers and more nuanced capability descriptions would require extending the metadata and cost function.
Finally, our predictor relies on simple features (token count and queue state) and may not capture content-dependent latency variations such as pathological prompts or tool-calling behavior.

\paragraph{Future work.}
We plan to explore (i) replicated controllers with consistent hashing for horizontal scalability, (ii) GPU-accelerated scheduling kernels for clusters with thousands of workers, (iii) richer feature sets that incorporate lightweight content features or request classes, and (iv) integration with agent frameworks that issue multi-step workflows rather than single LLM calls.

\section{Conclusion}

We have presented DIO, a predictive orchestrator for heterogeneous LLM inference clusters.
DIO operates as a model-agnostic control plane that learns per-worker performance online using a dual-timescale NLMS predictor and routes requests according to a tier- and Roofline-aware cost function that explicitly models VRAM headroom.

Our implementation in Go orchestrates Python-based vLLM workers over gRPC with only 14~$\mu$s median scheduling overhead.
Across 12 experiments on L4 and A100 hardware, including real production traces and stress tests, DIO reduces p99 latency by up to 63\% relative to Round-Robin baselines, eliminates OOM failures entirely, and maintains stable behavior under cold starts and overload.
These results suggest that lightweight control-theoretic scheduling, coupled with explicit memory modeling, is a practical path toward robust, cost-efficient LLM serving on heterogeneous fleets.
```

***

If you tell me which subsection you plan to edit first (e.g., “replace just 3.1 and 3.2.1 now”), I can further tailor those pieces or align them exactly with your current labels/figure numbers.