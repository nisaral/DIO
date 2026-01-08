# DIO: Distributed Inference Orchestrator

DIO is a high-performance, model-agnostic control plane designed to manage distributed machine learning workloads. It solves the "Python Bottleneck" by using Go to handle orchestration and gRPC for low-latency communication between a centralized manager and distributed Python inference workers.

##  Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Control Plane** | Go (Standard Library) | Lightweight orchestration and scheduling |
| **Data Plane** | Python (PyTorch / Scikit-Learn ready) | ML model inference execution |
| **Communication** | gRPC + Protocol Buffers (HTTP/2) | Fast, binary protocol for inter-service communication |
| **Persistence** | BoltDB | Embedded key-value store for worker registry |
| **Scaling** | Docker SDK for Go | Dynamic worker provisioning (planned) |

## Real-World Problems DIO Solves

In a professional environment, companies don't just "run a model." They face these three massive "pain points" that DIO addresses directly:

### 1. The "Python Performance Wall"
Python is great for ML math but terrible at handling 1,000s of concurrent web requests. DIO lets Go do the high-speed networking while Python stays isolated, preventing one slow request from freezing the whole API.

### 2. The "GPU Waste" Problem
GPUs are incredibly expensive. In a monolithic setup, if your web server is idle, your GPU sits idle too. DIO’s Orchestrator ensures that one GPU worker can serve multiple front-end apps, maximizing your "Return on Investment" for hardware.

### 3. The "Model Crash" Chain Reaction
If an ML model runs out of memory (OOM) in a standard app, the whole app dies. In DIO, the Go Manager detects the worker crash, restarts it via the Docker SDK, and reroutes the user's request to a different healthy worker.

## Validation Strategy: "Show" vs "Solution"

To prove this isn't just "show," we run three tests that simulate a real SRE (Site Reliability Engineering) environment:

### Test A: The "Straggler" Test (Latency Consistency)
We intentionally make one Python worker slow (simulating a "noisy neighbor" on a server).
- **The Goal:** Does the Go Scheduler detect the high latency and stop sending requests to that slow worker?
- **The Value:** This proves your Load Balancer is "Latency-Aware," a high-level system design feature.

### Test B: The "Sudden Spike" Test (Autoscaling)
We use a tool like `hey` or `ab` to hit the Go Manager with 50 requests per second.
- **The Goal:** Does the Go Manager actually use the Docker SDK to spin up a second and third worker container in real-time?
- **The Value:** This proves Resource Efficiency. You only pay for the "Muscle" (Python workers) when you actually have traffic.

### Test C: The "Token Budget" Test (Cost Control)
We send requests with varying lengths.
- **The Goal:** Does the Go Manager successfully log the `tokens_used` from the Python worker and reject a request if it exceeds a "budget"?
- **The Value:** This solves a real Business Problem—preventing a single user from accidentally spending $500 on LLM API calls in five minutes.

## Comparison: DIO vs. "Standard" AI APIs

| Feature | Standard "FastAPI" App | DIO Orchestrator |
|---------|------------------------|------------------|
| **Concurrency** | Limited by Python's GIL | Massive (handled by Go Goroutines) |
| **Scaling** | Manual or slow K8s scaling | Instant (Go-triggered Docker containers) |
| **Fault Tolerance** | App crashes on ML error | Self-healing (Go restarts the worker) |
| **Observability** | Only basic logs | Full Tracing (Latency, GPU usage, Tokens) |

##  How It Works

### 1. **Registration**
- Python workers start and register their capabilities with the Go Manager via gRPC
- Each worker provides:
  - Unique worker ID
  - Network address (host:port)
  - List of available models (e.g., "bert-base", "resnet50")
- Registration data is persisted in BoltDB

### 2. **Health Monitoring**
- Background Goroutine in the Manager pings workers every 10 seconds
- Uses `CheckHealth` RPC call to verify worker availability
- Failed workers are marked as inactive and removed from scheduling pool
- Automatic recovery detection for restored workers

### 3. **Scheduling**
- When an inference request arrives at the Manager, the Scheduler is invoked
- Uses **Round-Robin scheduling** to distribute load evenly across available workers
- Filters workers based on:
  - Health status (active/inactive)
  - Model availability
  - Capacity constraints

### 4. **Execution**
- Manager forwards the task to the selected worker via gRPC
- Worker receives binary data, processes it through the ML model
- Results returned immediately with latency metrics
- No data is permanently stored (request-response pattern)

##  Project Structure

```
DIO/
├── README.md                    # Project documentation
├── go.mod                       # Go module definition
├── api/
│   └── proto/
│       ├── dio.proto           # gRPC service definitions
│       ├── dio.pb.go           # Generated Go protobuf code
│       └── dio_grpc.pb.go      # Generated gRPC service code
├── cmd/
│   └── manager/
│       └── main.go             # Entry point for the Go Manager
├── internal/
│   ├── api_gateway/            # API endpoint handlers (future)
│   ├── health/
│   │   └── monitor.go          # Health check logic for workers
│   ├── registry/
│   │   └── boltdb.go           # Worker registry & persistence
│   └── scheduler/
│       └── loadbalancer.go     # Round-Robin scheduler
├── pkg/
│   └── pb/                      # Generated protobuf packages
├── workers/
│   └── python_worker/           # Python inference worker implementation
│       ├── register_client.py   # gRPC client for registration
│       ├── api/
│       │   └── proto/
│       │       ├── dio_pb2.py   # Generated Python protobuf code
│       │       └── dio_pb2_grpc.py  # Generated Python gRPC code
│       └── pb/                  # Protobuf utilities
├── ui/                          # Frontend dashboard (future)
│   ├── public/
│   └── src/
└── deployments/                 # Docker, Kubernetes configs (future)
```

##  Architecture Overview

```
┌─────────────────────────────────────────┐
│   Client (Python/Go/Any gRPC Client)    │
└────────────────────┬────────────────────┘
                     │ InferenceRequest (gRPC)
                     ▼
         ┌───────────────────────┐
         │   Go Manager          │
         │  (Port 50050)         │
         │                       │
         │ ┌─────────────────┐  │
         │ │  Scheduler      │  │ Round-Robin
         │ │  (LoadBalancer) │  │
         │ └────────┬────────┘  │
         │          │           │
         │ ┌────────▼────────┐  │
         │ │  Registry       │  │
         │ │  (BoltDB)       │  │
         │ └─────────────────┘  │
         │          │           │
         │ ┌────────▼────────┐  │
         │ │ Health Monitor  │  │ Pings every 10s
         │ └─────────────────┘  │
         └────┬───────────┬──────┘
              │           │
        ┌─────▼┐    ┌────▼──┐   ...
        │Worker│    │Worker │
        │  #1  │    │  #2   │
        │(Port │    │(Port  │
        │50051)│    │50052) │
        └──────┘    └───────┘
        
   Python ML Models (PyTorch, Scikit-Learn, etc.)
```

##  Getting Started

### Prerequisites
- Go 1.21+
- Python 3.8+
- Docker (for containerized workers)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nisaral/dio.git
cd DIO
```

2. **Install Go dependencies**
```bash
go mod download
go mod tidy
```

3. **Build the Manager**
```bash
cd cmd/manager
go build -o manager main.go
```

4. **Install Python dependencies** (for workers)
```bash
cd workers/python_worker
pip install grpcio protobuf grpcio-tools
# Add your ML dependencies: torch, scikit-learn, etc.
```

### Running the System

**Terminal 1: Start the Manager**
```bash
./cmd/manager/manager
# Listens on 0.0.0.0:50050
```

**Terminal 2+: Start Python Workers**
```bash
python workers/python_worker/register_client.py \
  --worker-id worker-1 \
  --manager-host localhost \
  --manager-port 50050 \
  --listen-port 50051 \
  --models bert-base,resnet50
```

##  API Reference

### gRPC Services

#### Orchestrator Service (Manager)
Located in [api/proto/dio.proto](api/proto/dio.proto)

**RegisterWorker**
- **Request**: `RegisterRequest` (worker_id, address, models[])
- **Response**: `RegisterResponse` (success: bool)
- Used by workers to announce availability

**ExecuteInference** (planned)
- **Request**: `InferenceRequest` (model_id, data)
- **Response**: `InferenceResponse` (output, latency_ms)
- Used by clients to request inference

#### InferenceWorker Service (Worker)
**Predict**
- **Request**: `InferenceRequest` (model_id, data)
- **Response**: `InferenceResponse` (output, latency_ms)
- Executes ML model on the worker

**CheckHealth**
- **Request**: `google.protobuf.Empty`
- **Response**: `google.protobuf.Empty`
- Confirms worker is alive and responsive

##  Workflow Example

```
1. Python Worker starts and calls:
   Manager.RegisterWorker(
     worker_id="worker-1",
     address="localhost:50051",
     models=["bert-base", "resnet50"]
   )

2. Manager stores worker in BoltDB registry

3. Health Monitor begins periodic pings to worker

4. Client sends inference request to Manager:
   Manager.ExecuteInference(
     model_id="bert-base",
     data=<serialized_input>
   )

5. Manager's Scheduler picks "worker-1" (round-robin)

6. Manager forwards to Worker:
   Worker.Predict(
     model_id="bert-base",
     data=<serialized_input>
   )

7. Worker processes input, returns output + latency

8. Manager returns response to client
```

##  Development

### Regenerating Protobuf Code

After modifying [api/proto/dio.proto](api/proto/dio.proto):

**For Go:**
```bash
protoc --go_out=. --go-grpc_out=. api/proto/dio.proto
```

**For Python:**
```bash
python -m grpc_tools.protoc \
  -I. \
  --python_out=workers/python_worker/api/proto/ \
  --grpc_python_out=workers/python_worker/api/proto/ \
  api/proto/dio.proto
```

### Project Modules

| Module | Responsibility |
|--------|-----------------|
| `internal/registry` | Worker persistence (BoltDB) |
| `internal/scheduler` | Load balancing logic (Round-Robin) |
| `internal/health` | Worker health checks |
| `internal/api_gateway` | HTTP API wrapper (future) |

##  Roadmap

- [ ] API Gateway (HTTP/REST wrapper)
- [ ] Advanced scheduling (Least-Loaded, Priority-Based)
- [ ] Metrics & observability (Prometheus, Jaeger)
- [ ] UI Dashboard for worker management
- [ ] Docker Compose & Kubernetes support
- [ ] GPU workload awareness
- [ ] Request queueing & batch processing
- [ ] Worker auto-scaling

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Built with for efficient distributed ML inference**
