package health

import (
	"context"
	"log"
	"time"

	"github.com/nisaral/dio/internal/registry"
	pb "github.com/nisaral/dio/api/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// StartMonitor runs a background loop to check worker health
func StartMonitor(store *registry.Store, interval time.Duration) {
	ticker := time.NewTicker(interval)
	go func() {
		for range ticker.C {
			workers, err := store.ListWorkers()
			if err != nil {
				log.Printf("Monitor error: %v", err)
				continue
			}

			for _, w := range workers {
				checkWorker(w)
			}
		}
	}()
}

func checkWorker(w *pb.RegisterRequest) {
	// Connect to the Python Worker's gRPC server
	conn, err := grpc.Dial(w.Address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("Worker %s is UNREACHABLE", w.WorkerId)
		return
	}
	defer conn.Close()

	client := pb.NewInferenceWorkerClient(conn)
	
	// Set a strict 2-second timeout for the health check
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err = client.Predict(ctx, &pb.InferenceRequest{})
	if err != nil {
		log.Printf("Worker %s FAILED health check: %v", w.WorkerId, err)
		// Logic for Version 2: Remove from registry if it fails X times
	} else {
		log.Printf("Worker %s is HEALTHY", w.WorkerId)
	}
}