package main

import (
	"context"
	"encoding/json"
	"fmt"
	"encoding/csv" // New: for logging
	"sync" // New: for thread-safe file writing
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
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
type TelemetryLogger struct {
    mu     sync.Mutex
    file   *os.File
    writer *csv.Writer
}

func NewTelemetryLogger(filename string) (*TelemetryLogger, error) {
	if err := os.MkdirAll(filepath.Dir(filename), 0755); err != nil {
		return nil, fmt.Errorf("failed to create dir: %v", err)
	}

	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, err
	}
	writer := csv.NewWriter(f)
	info, _ := f.Stat()
	if info.Size() == 0 {
		writer.Write([]string{"timestamp", "worker_id", "status", "latency_ms", "ttft_ms", "tokens", "tpot"})
		writer.Flush()
	}
	return &TelemetryLogger{file: f, writer: writer}, nil
}

func (tl *TelemetryLogger) Log(workerID, status string, m scheduler.DetailedMetrics, tokens int) {
	tl.mu.Lock()
	defer tl.mu.Unlock()
	row := []string{
		time.Now().Format(time.RFC3339),
		workerID,
		status,
		fmt.Sprintf("%.2f", m.E2ELatency),
		fmt.Sprintf("%.2f", m.TTFT),
		strconv.Itoa(tokens),
		fmt.Sprintf("%.2f", m.TPOT),
	}
	tl.writer.Write(row)
	tl.writer.Flush()
}

type server struct {
	pb.UnimplementedOrchestratorServer
	store     *registry.Store
	scheduler *scheduler.Scheduler
	logger    *TelemetryLogger
}

func (s *server) ExecuteInference(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
    workerID, err := s.scheduler.PickBestWorker(req)
    if err != nil {
        return nil, err
    }

    worker, ok := s.scheduler.GetWorker(workerID)
    if !ok {
        return nil, fmt.Errorf("worker not found")
    }

    conn, err := grpc.NewClient(worker.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        return nil, err
    }
    defer conn.Close()

    ctx, cancel := context.WithTimeout(ctx, 300*time.Second) // Long timeout for L4/CPU
    defer cancel()

    workerClient := pb.NewInferenceWorkerClient(conn)
    resp, err := workerClient.Predict(ctx, req)
    
    inputTokens := len(req.Data) / 4
    metrics := scheduler.DetailedMetrics{}

    if err == nil {
        metrics = scheduler.DetailedMetrics{
            TTFT:       float64(resp.TtftMs),
            E2ELatency: float64(resp.LatencyMs),
            // Calculate TPOT inside manager to be safe
        }
        if resp.TokensUsed > 1 {
            metrics.TPOT = (metrics.E2ELatency - metrics.TTFT) / float64(resp.TokensUsed-1)
        }
        
        s.scheduler.FeedbackLoop(workerID, metrics, inputTokens)
        s.logger.Log(workerID, "SUCCESS", metrics, int(resp.TokensUsed)) // LOG SUCCESS
    } else {
        log.Printf("Worker %s failed: %v", workerID, err)
        s.scheduler.FeedbackLoop(workerID, scheduler.DetailedMetrics{E2ELatency: -1}, 0)
        s.logger.Log(workerID, "FAILED", scheduler.DetailedMetrics{E2ELatency: -1}, 0) // LOG FAILURE
        return nil, err
    }

    return resp, nil
} 
func (s *server) RegisterWorker(ctx context.Context, req *pb.RegisterRequest) (*pb.RegisterResponse, error) {
	log.Printf("Registering worker: %s at %s", req.WorkerId, req.Address)
	if err := s.store.SaveWorker(req); err != nil {
		return &pb.RegisterResponse{Success: false}, err
	}
	// VRAM is passed as MB from Python workers
	s.scheduler.RegisterWorker(req.WorkerId, req.Address, req.Tier, int64(req.VramGb))
	return &pb.RegisterResponse{Success: true}, nil
}



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

		ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
		defer cancel()

		tier := payload["tier"]
		if tier == "" {
			tier = "small"
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
        log.Printf("DEBUG API: Current worker list requested. Found: %v", workers)
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(map[string]interface{}{
            "workers":      workers,
            "worker_count": len(workers), // Add this explicitly
        })
    }
}

func main() {

	telemetryFile := os.Getenv("TELEMETRY_FILE")
	if telemetryFile == "" {
		telemetryFile = "benchmarks/results_cloud/manager_telemetry.csv"
	}

	telemetry, err := NewTelemetryLogger(telemetryFile)
	if err != nil {
		log.Fatalf("Failed to init telemetry: %v", err)
	}

	// 1. Initialize Telemetry Logger


	store, err := registry.NewStore("dio_registry.db")
	if err != nil {
		log.Fatalf("Failed to init store: %v", err)
	}

	sched := scheduler.NewScheduler()
	if strategy := os.Getenv("SCHEDULER_STRATEGY"); strategy != "" {
		sched.Strategy = strategy
		log.Printf("🔧 Scheduler Strategy set to: %s", strategy)
	}

	// 1. Setup gRPC Listener
	lis, err := net.Listen("tcp", "0.0.0.0:50055")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterOrchestratorServer(grpcServer, &server{
		store:     store,
		scheduler: sched,
		logger:    telemetry, // Pass to server
	})

	// 2. Start gRPC server in background
	go func() {
		log.Printf("DIO Manager gRPC listening at %v", lis.Addr())
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("Failed to serve gRPC: %v", err)
		}
	}()

	// 3. Give gRPC a moment to bind before localClient dials
	time.Sleep(2 * time.Second)

	// 4. Setup Local gRPC Client for HTTP Gateway
	localConn, err := grpc.NewClient("127.0.0.1:50055", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("Failed to create local client: %v", err)
	}
	defer localConn.Close()
	localClient := pb.NewOrchestratorClient(localConn)

	// 5. Setup HTTP Routes
	gateway := api_gateway.NewAPIGateway("localhost:50055", sched)
	http.HandleFunc("/api/test", enableCORS(gateway.HandleTest))
	http.HandleFunc("/api/generate", enableCORS(handleGenerate(localClient)))
	http.HandleFunc("/debug/reset_worker", enableCORS(handleDebugReset(sched)))
	http.HandleFunc("/debug/prediction", enableCORS(handleDebugPrediction(sched)))
	http.HandleFunc("/debug/workers", enableCORS(handleDebugListWorkers(sched)))
	http.Handle("/", http.FileServer(http.Dir("./ui/src")))

	log.Printf("DIO Manager HTTP API listening at :8085")

	// 6. Optional Autoscaler
	dockerMgr, err := worker_mgmt.NewDockerManager()
	if err == nil {
		go scheduler.StartAutoscaler(sched, dockerMgr, 2)
	}

	// 7. Start HTTP Server (Blocking)
	if err := http.ListenAndServe(":8085", nil); err != nil {
		log.Fatalf("HTTP server failed: %v", err)
	}
}
