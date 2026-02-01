package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	pb "github.com/nisaral/dio/api/proto"
	"github.com/nisaral/dio/internal/api_gateway"
	"github.com/nisaral/dio/internal/registry"
	"github.com/nisaral/dio/internal/scheduler"
	"github.com/nisaral/dio/workers/worker_mgmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

type server struct {
	pb.UnimplementedOrchestratorServer
	store     *registry.Store
	scheduler *scheduler.Scheduler
}

func (s *server) RegisterWorker(ctx context.Context, req *pb.RegisterRequest) (*pb.RegisterResponse, error) {
	log.Printf("Registering worker: %s at %s", req.WorkerId, req.Address)
	if err := s.store.SaveWorker(req); err != nil {
		return &pb.RegisterResponse{Success: false}, err
	}
	s.scheduler.RegisterWorker(req.WorkerId, req.Address, req.Tier, int64(req.VramGb))
	return &pb.RegisterResponse{Success: true}, nil
}

// Inside your Server struct methods...

func (s *server) ExecuteInference(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	// 2. Scheduler Picks Worker (SJF)
	workerID, err := s.scheduler.PickBestWorker(req)
	if err != nil {
		return nil, err
	}

	// Send the selected WorkerID back to the caller (HTTP handler) via gRPC Header
	grpc.SetHeader(ctx, metadata.Pairs("x-dio-worker-id", workerID))

	// 3. Execute gRPC
	// Retrieve worker address from scheduler
	worker, ok := s.scheduler.GetWorker(workerID)
	if !ok {
		return nil, fmt.Errorf("worker %s not found after selection", workerID)
	}

	// Establish gRPC connection to the worker
	conn, err := grpc.NewClient(worker.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to worker %s at %s: %v", workerID, worker.Address, err)
	}
	defer conn.Close()

	// Enforce a timeout for the gRPC call to fail fast if worker is unreachable
	ctx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()

	workerClient := pb.NewInferenceWorkerClient(conn)
	resp, err := workerClient.Predict(ctx, req)
	if err != nil {
		log.Printf("Error calling worker %s: %v", workerID, err)
	}

	// 4. Feedback Loop (The Critical Research Step)
	// Calculate input tokens (approx)
	inputTokens := len(req.Data) / 4
	// Be safe if resp is nil on error
	if err == nil {
		// Calculate Detailed Metrics for Research Paper
		ttft := float64(resp.GetTtftMs())
		e2e := float64(resp.LatencyMs)
		outputTokens := float64(resp.TokensUsed)

		tpot := 0.0
		if outputTokens > 1 {
			tpot = (e2e - ttft) / (outputTokens - 1)
		}

		inputTPS := 0.0
		if ttft > 0 {
			inputTPS = float64(inputTokens) / (ttft / 1000.0)
		}

		outputTPS := 0.0
		if e2e > ttft {
			outputTPS = (outputTokens - 1) / ((e2e - ttft) / 1000.0)
		}

		metrics := scheduler.DetailedMetrics{
			TTFT:             ttft,
			TPOT:             tpot,
			E2ELatency:       e2e,
			InputThroughput:  inputTPS,
			OutputThroughput: outputTPS,
		}
		s.scheduler.FeedbackLoop(workerID, metrics, inputTokens)
	} else {
		// Even on error, we must decrement the pending task count!
		// We pass -1 latency so it doesn't skew the predictor
		s.scheduler.FeedbackLoop(workerID, scheduler.DetailedMetrics{E2ELatency: -1}, 0)
		return nil, err
	}

	return resp, nil
}

// enableCORS wraps an http.HandlerFunc to allow cross-origin requests (e.g., from VS Code Live Server)
func enableCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
		if r.Method == "OPTIONS" {
			return
		}
		next(w, r)
	}
}

// handleGenerate provides a standard JSON endpoint for benchmarking tools like Locust
func handleGenerate(client pb.OrchestratorClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, _ := io.ReadAll(r.Body)
		var payload map[string]string
		if err := json.Unmarshal(body, &payload); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		// Add a timeout to prevent infinite hanging if workers are unreachable
		ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
		defer cancel()

		// Propagate Tier from HTTP request
		tier := payload["tier"]
		if tier == "" {
			tier = "small" // Default
		}

		var header metadata.MD
		resp, err := client.ExecuteInference(ctx, &pb.InferenceRequest{
			ModelId: payload["model_id"],
			Data:    []byte(payload["prompt"]),
			Tier:    tier,
		}, grpc.Header(&header))

		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		// Expose Worker ID to client for benchmarking
		if workers := header.Get("x-dio-worker-id"); len(workers) > 0 {
			w.Header().Set("X-DIO-Worker-ID", workers[0])
		}

		json.NewEncoder(w).Encode(resp)
	}
}

func handleDebugReset(sched *scheduler.Scheduler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var payload map[string]string
		json.NewDecoder(r.Body).Decode(&payload)
		workerID := payload["worker_id"]
		sched.ResetWorkerState(workerID)
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("Worker state reset"))
	}
}

func handleDebugPrediction(sched *scheduler.Scheduler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		workerID := r.URL.Query().Get("worker")
		tokens, _ := strconv.Atoi(r.URL.Query().Get("tokens"))
		if tokens == 0 {
			tokens = 50
		}
		pred := sched.GetDebugPrediction(workerID, tokens)
		json.NewEncoder(w).Encode(map[string]float64{"predicted_ms": pred})
	}
}

func handleDebugListWorkers(sched *scheduler.Scheduler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		workers := sched.ListWorkers()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string][]string{
			"workers": workers,
		})
	}
}

func main() {
	store, err := registry.NewStore("dio_registry.db")
	if err != nil {
		log.Fatalf("Failed to init store: %v", err)
	}

	sched := scheduler.NewScheduler()
	// Configure Scheduler Strategy from Env (for Baselines)
	if strategy := os.Getenv("SCHEDULER_STRATEGY"); strategy != "" {
		sched.Strategy = strategy
		log.Printf("🔧 Scheduler Strategy set to: %s", strategy)
	}

	// gRPC server (port 50052)
	lis, err := net.Listen("tcp", ":50052")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterOrchestratorServer(grpcServer, &server{
		store:     store,
		scheduler: sched,
	})

	// Create a local client for the HTTP handlers to use
	localConn, err := grpc.NewClient("localhost:50052", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("Failed to create local client: %v", err)
	}
	localClient := pb.NewOrchestratorClient(localConn)

	// HTTP API Gateway (port 8080)
	gateway := api_gateway.NewAPIGateway("localhost:50052", sched)
	http.HandleFunc("/api/test", enableCORS(gateway.HandleTest))
	http.HandleFunc("/api/generate", enableCORS(handleGenerate(localClient)))
	http.HandleFunc("/debug/reset_worker", enableCORS(handleDebugReset(sched)))
	http.HandleFunc("/debug/prediction", enableCORS(handleDebugPrediction(sched)))
	http.HandleFunc("/debug/workers", enableCORS(handleDebugListWorkers(sched)))
	http.Handle("/", http.FileServer(http.Dir("./ui/src")))

	log.Printf("DIO Manager gRPC listening at %v", lis.Addr())
	log.Printf("DIO Manager HTTP API listening at :8080")

	// Initialize Docker Manager for autoscaling
	dockerMgr, err := worker_mgmt.NewDockerManager()
	if err != nil {
		log.Printf("Warning: Docker not found, autoscaling disabled: %v", err)
	} else {
		go scheduler.StartAutoscaler(sched, dockerMgr, 2)
	}

	// Start HTTP server in goroutine
	go func() {
		if err := http.ListenAndServe(":8080", nil); err != nil {
			log.Fatalf("HTTP server failed: %v", err)
		}
	}()

	// Start gRPC server (blocking)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve gRPC: %v", err)
	}
}
