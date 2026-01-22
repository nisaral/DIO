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
	return &pb.RegisterResponse{Success: true}, nil
}

func (s *server) ExecuteInference(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	workerList, _ := s.store.ListWorkers()

	if len(workerList) == 0 {
		log.Printf("No workers available, returning mock response")
		// Return a mock response for testing without workers
		return &pb.InferenceResponse{ // Corrected: InferenceResponse does not have a Success field.
			Output:     []byte("Mock response from manager (no workers available)"),
			TokensUsed: 0,
		}, nil
	}

	// Convert pb.RegisterRequest to registry.Worker
	var workers []*registry.Worker
	for _, w := range workerList {
		workers = append(workers, &registry.Worker{
			ID:        w.WorkerId,
			Address:   w.Address,
			IsHealthy: true,
		})
	}

	s.scheduler.UpdateWorkers(workers)

	// Estimate tokens (heuristic: 1 token approx 4 bytes)
	tokens := len(req.Data) / 4
	if tokens == 0 {
		tokens = 1
	}

	target := s.scheduler.PickBestWorker(tokens)
	if target == nil {
		return nil, fmt.Errorf("no workers available")
	}

	conn, err := grpc.Dial(target.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	// Queue Awareness: Track Queue Depth
	// We push to the channel to signal "Busy/Queued" and defer the pop
	select {
	case target.TaskChannel <- req:
		defer func() { <-target.TaskChannel }()
	default:
		// Queue is full (Backpressure). For now, we block until a slot opens.
		target.TaskChannel <- req
		defer func() { <-target.TaskChannel }()
	}

	start := time.Now()
	client := pb.NewInferenceWorkerClient(conn)
	resp, err := client.Predict(ctx, req)

	if err == nil {
		// Feedback Loop: Update RLS predictor with actual latency
		s.scheduler.FeedbackLoop(target.ID, float64(time.Since(start).Milliseconds()), int(resp.TokensUsed))
	}

	return resp, err
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

		resp, err := client.ExecuteInference(r.Context(), &pb.InferenceRequest{
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
		scheduler.StartAutoscaler(sched, dockerMgr, 2)
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
