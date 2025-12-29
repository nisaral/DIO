package main

import (
	"context"
	"fmt"
	"log"
	"time"

	pb "github.com/nisaral/dio/api/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	// Connect to the Go Manager
	conn, err := grpc.Dial("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	client := pb.NewOrchestratorClient(conn)

	// Simulate an inference request
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	resp, err := client.ExecuteInference(ctx, &pb.InferenceRequest{
		ModelId: "fraud-detection",
		Data:    []byte("sample_transaction_data"),
	})

	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	fmt.Printf("Result from DIO: %s (Latency: %.2fms)\n", string(resp.Output), resp.LatencyMs)
}