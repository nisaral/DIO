package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/nisaral/dio/api/proto"
)

var grpcClient pb.OrchestratorClient

func main() {
	port := flag.String("port", "8080", "Dashboard HTTP port")
	managerAddr := flag.String("manager", "localhost:50051", "DIO Manager gRPC address")
	flag.Parse()

	// Connect to Manager
	conn, err := grpc.NewClient(*managerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to manager: %v", err)
	}
	defer conn.Close()
	grpcClient = pb.NewOrchestratorClient(conn)

	// Setup HTTP Server
	http.Handle("/", http.FileServer(http.Dir("../../ui/public")))
	http.HandleFunc("/api/test", handleTestRequest)

	fmt.Printf("DIO Dashboard running at http://localhost:%s\n", *port)
	log.Fatal(http.ListenAndServe(":"+*port, nil))
}

type TestResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
	Latency int64  `json:"latency_ms"`
	Tokens  int64  `json:"tokens"`
}

func handleTestRequest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Type    string `json:"type"`
		ModelID string `json:"model_id"`
		Payload string `json:"payload"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	start := time.Now()

	// Map UI request to gRPC request
	grpcReq := &pb.InferenceRequest{
		ModelId: req.ModelID,
		Data:    []byte(req.Payload),
	}

	resp, err := grpcClient.ExecuteInference(ctx, grpcReq)

	w.Header().Set("Content-Type", "application/json")
	jsonResp := TestResponse{
		Latency: time.Since(start).Milliseconds(),
	}

	if err != nil {
		jsonResp.Success = false
		jsonResp.Message = fmt.Sprintf("Error: %v", err)
	} else {
		jsonResp.Success = true
		jsonResp.Message = "Inference executed successfully"
		// Assuming TokensUsed is part of the response based on context
		jsonResp.Tokens = int64(resp.TokensUsed)
	}

	json.NewEncoder(w).Encode(jsonResp)
}
