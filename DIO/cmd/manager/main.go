package main

import (
	"context"
	"fmt"
	"log"
	"net"

	"github.com/nisaral/dio/internal/registry"
	"github.com/nisaral/dio/internal/scheduler" // Added this
   pb "github.com/nisaral/dio/api/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure" // Added for gRPC Dial
)

// server now has the scheduler field
type server struct {
	pb.UnimplementedOrchestratorServer
	store     *registry.Store
	scheduler *scheduler.Scheduler // Field added here
}

func (s *server) RegisterWorker(ctx context.Context, req *pb.RegisterRequest) (*pb.RegisterResponse, error) {
	log.Printf("Registering worker: %s at %s", req.WorkerId, req.Address)
	if err := s.store.SaveWorker(req); err != nil {
		return &pb.RegisterResponse{Success: false}, err
	}
	return &pb.RegisterResponse{Success: true}, nil
}

func (s *server) ExecuteInference(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	// 1. Refresh scheduler with latest workers from DB
	workerList, _ := s.store.ListWorkers()
	s.scheduler.UpdateWorkers(workerList)

	// 2. Pick a worker using Round-Robin
	target := s.scheduler.PickWorker()
	if target == nil {
		return nil, fmt.Errorf("no workers available")
	}

	// 3. Connect to the Python Worker
	conn, err := grpc.Dial(target.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	client := pb.NewInferenceWorkerClient(conn)
	return client.Predict(ctx, req)
}

func main() {
	store, err := registry.NewStore("dio_registry.db")
	if err != nil {
		log.Fatalf("Failed to init store: %v", err)
	}

	// Initialize the scheduler
	sched := scheduler.NewScheduler()

	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	s := grpc.NewServer()
	// Pass both store and scheduler to the server
	pb.RegisterOrchestratorServer(s, &server{
		store:     store,
		scheduler: sched,
	})

	log.Printf("DIO Manager listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}