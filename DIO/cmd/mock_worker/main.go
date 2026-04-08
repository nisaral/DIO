package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net"
	"time"

	pb "github.com/nisaral/dio/api/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/emptypb"
)

// mockWorkerServer implements the InferenceWorker gRPC service
type mockWorkerServer struct {
	pb.UnimplementedInferenceWorkerServer
	workerID    string
	latencyMult float64
	baseLatency float64
}

func (s *mockWorkerServer) Predict(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	// Simulate latency based on input size + multiplier
	dataLen := len(req.Data)
	tokens := dataLen / 4 // rough approximation
	if tokens < 10 {
		tokens = 10
	}
	if tokens > 500 {
		tokens = 500
	}

	// Base latency + per-token cost + random jitter
	baseMs := s.baseLatency
	perTokenMs := 0.8 * s.latencyMult
	jitterMs := (rand.Float64() - 0.5) * 20.0 // ±10ms jitter

	totalMs := (baseMs + float64(tokens)*perTokenMs + jitterMs) * s.latencyMult
	if totalMs < 5 {
		totalMs = 5
	}

	// TTFT is typically ~20-40% of total latency
	ttftMs := totalMs * (0.2 + rand.Float64()*0.2)

	// Simulate the processing time
	sleepMs := totalMs
	if sleepMs > 2000 {
		sleepMs = 2000 // cap sleep for demo speed
	}
	time.Sleep(time.Duration(sleepMs) * time.Millisecond)

	output := fmt.Sprintf("Mock response from worker %s: processed %d tokens in %.1fms", s.workerID, tokens, totalMs)

	return &pb.InferenceResponse{
		Output:     []byte(output),
		LatencyMs:  float32(totalMs),
		TokensUsed: int64(tokens),
		TtftMs:     float32(ttftMs),
	}, nil
}

func (s *mockWorkerServer) CheckHealth(ctx context.Context, _ *emptypb.Empty) (*emptypb.Empty, error) {
	return &emptypb.Empty{}, nil
}

func main() {
	workerID := flag.String("worker-id", "mock-w1", "Worker ID")
	port := flag.Int("port", 50060, "gRPC listen port")
	vram := flag.Int("vram", 4000, "VRAM in MB to report")
	managerAddr := flag.String("manager-addr", "localhost:50055", "DIO Manager gRPC address")
	latencyMult := flag.Float64("latency-mult", 1.0, "Latency multiplier (1.0 = normal speed)")
	mock := flag.Bool("mock", true, "Mock mode (no real model)")
	flag.Parse()

	_ = *mock // always mock in this binary

	log.SetFlags(log.Ltime | log.Lshortfile)
	log.Printf("[MockWorker] Starting %s on port %d (latency_mult=%.1f, vram=%d MB)", *workerID, *port, *latencyMult, *vram)

	// Start gRPC server
	lis, err := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", *port))
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterInferenceWorkerServer(grpcServer, &mockWorkerServer{
		workerID:    *workerID,
		latencyMult: *latencyMult,
		baseLatency: 15.0, // 15ms base latency
	})

	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("gRPC serve failed: %v", err)
		}
	}()

	// Register with DIO Manager
	time.Sleep(500 * time.Millisecond) // wait for server to start
	log.Printf("[MockWorker] Registering with manager at %s...", *managerAddr)

	conn, err := grpc.NewClient(*managerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to manager: %v", err)
	}
	defer conn.Close()

	client := pb.NewOrchestratorClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	resp, err := client.RegisterWorker(ctx, &pb.RegisterRequest{
		WorkerId: *workerID,
		Address:  fmt.Sprintf("localhost:%d", *port),
		Models:   []string{"demo-model"},
		Tier:     "small",
		VramGb:   int64(*vram),
	})
	if err != nil {
		log.Fatalf("Registration failed: %v", err)
	}
	if !resp.Success {
		log.Fatal("Registration rejected by manager")
	}

	log.Printf("[MockWorker] ✓ Registered successfully as %s on port %d", *workerID, *port)
	log.Printf("[MockWorker] Ready to serve predictions (latency_mult=%.1f)", *latencyMult)

	// Keep alive
	select {}
}
