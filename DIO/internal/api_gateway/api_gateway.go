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
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight requests
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req TestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Connect to gRPC manager
	conn, err := grpc.Dial(ag.managerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		respondJSON(w, http.StatusServiceUnavailable, TestResponse{
			Success: false,
			Message: "Failed to connect to manager: " + err.Error(),
		})
		return
	}
	defer conn.Close()

	client := pb.NewOrchestratorClient(conn)

	// Measure latency
	start := time.Now()
	resp, err := client.ExecuteInference(ctx, &pb.InferenceRequest{
		ModelId: req.ModelID,
		Data:    []byte(req.Payload),
	})
	latency := time.Since(start).Milliseconds()

	if err != nil {
		log.Printf("Inference failed: %v", err)
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
		WorkerID:  "",
	})
}

func respondJSON(w http.ResponseWriter, statusCode int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(data)
}
