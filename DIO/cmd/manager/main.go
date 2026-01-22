package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"time"

	pb "github.com/nisaral/dio/api/proto"
	"github.com/nisaral/dio/internal/api_gateway"
	"github.com/nisaral/dio/internal/registry"
	"github.com/nisaral/dio/internal/scheduler"
	"github.com/nisaral/dio/workers/worker_mgmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
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
	s.scheduler.RegisterWorker(req.WorkerId, req.Address)
	return &pb.RegisterResponse{Success: true}, nil
}

// Inside your Server struct methods...

func (s *server) ExecuteInference(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	// 1. Start Timer
	start := time.Now()

	// 2. Scheduler Picks Worker (SJF)
	workerID, err := s.scheduler.PickBestWorker(req)
	if err != nil {
		return nil, err
	}

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
	// Capture latency with microsecond precision (e.g. 12.345 ms)
	duration := float64(time.Since(start).Microseconds()) / 1000.0

	// Calculate input tokens (approx)
	inputTokens := len(req.Data) / 4
	// Be safe if resp is nil on error
	if err == nil {
		s.scheduler.FeedbackLoop(workerID, duration, inputTokens)
	} else {
		// Even on error, we must decrement the pending task count!
		// We pass -1 latency so it doesn't skew the predictor
		s.scheduler.FeedbackLoop(workerID, -1, 0)
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

		resp, err := client.ExecuteInference(ctx, &pb.InferenceRequest{
			ModelId: payload["model_id"],
			Data:    []byte(payload["prompt"]),
		})

		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func main() {
	store, err := registry.NewStore("dio_registry.db")
	if err != nil {
		log.Fatalf("Failed to init store: %v", err)
	}

	sched := scheduler.NewScheduler()

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
