package api_gateway

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"time"

	pb "github.com/nisaral/dio/api/proto"
	"github.com/nisaral/dio/internal/scheduler"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type TestRequest struct {
	Type    string `json:"type"`
	ModelID string `json:"model_id"`
	Payload string `json:"payload"`
}

type TestResponse struct {
	Success   bool   `json:"success"`
	Message   string `json:"message,omitempty"`
	LatencyMs int64  `json:"latency_ms,omitempty"`
	Tokens    int    `json:"tokens,omitempty"`
	WorkerID  string `json:"worker_id,omitempty"`
}

type APIGateway struct {
	managerAddr string
	scheduler   *scheduler.Scheduler
}

func NewAPIGateway(managerAddr string, sched *scheduler.Scheduler) *APIGateway {
	return &APIGateway{
		managerAddr: managerAddr,
		scheduler:   sched,
	}
}

func (ag *APIGateway) HandleTest(w http.ResponseWriter, r *http.Request) {
	// 1. Enable CORS for local testing and UI
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	// 2. HEALTH CHECK (GET): Used by run_cloud_suite.py to verify manager is UP
	if r.Method == http.MethodGet {
		respondJSON(w, http.StatusOK, map[string]string{
			"status": "online",
			"time":   time.Now().Format(time.RFC3339),
		})
		return
	}

	// 3. INFERENCE TEST (POST): Used for manual gRPC verification
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req TestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Connect to gRPC manager (Ideally use a shared client, but keeping for compatibility)
	conn, err := grpc.NewClient(ag.managerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		respondJSON(w, http.StatusServiceUnavailable, TestResponse{
			Success: false,
			Message: "Failed to connect to gRPC: " + err.Error(),
		})
		return
	}
	defer conn.Close()

	client := pb.NewOrchestratorClient(conn)

	// Measure round-trip gRPC latency
	start := time.Now()
	resp, err := client.ExecuteInference(ctx, &pb.InferenceRequest{
		ModelId: req.ModelID,
		Data:    []byte(req.Payload),
		Tier:    "small", // Default for test
	})
	latency := time.Since(start).Milliseconds()

	if err != nil {
		log.Printf("[GATEWAY] Inference failed: %v", err)
		respondJSON(w, http.StatusInternalServerError, TestResponse{
			Success: false,
			Message: "Inference failed: " + err.Error(),
		})
		return
	}

	respondJSON(w, http.StatusOK, TestResponse{
		Success:   true,
		LatencyMs: latency,
		Tokens:    int(resp.TokensUsed),
		WorkerID:  "selected-by-scheduler",
	})
}

func respondJSON(w http.ResponseWriter, statusCode int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(data)
}