package scheduler

import (
	"context"
	"log"
	"time"
	"github.com/nisaral/dio/workers/worker_mgmt"
)

func StartAutoscaler(s *Scheduler, dm *worker_mgmt.DockerManager, threshold int) {
	ticker := time.NewTicker(15 * time.Second)
	go func() {
		for range ticker.C {
			s.mu.Lock()
			workerCount := len(s.workers)
			s.mu.Unlock()

			// If we have 0 workers or if they are all busy, spawn one
			if workerCount < threshold {
				log.Println("[Autoscaler] Demand high. Spawning new Python worker...")
				ctx := context.Background()
				
				// Ensure you have a docker image named 'dio-python-worker' built
				err := dm.SpawnWorker(ctx, "dio-python-worker:latest")
				if err != nil {
					log.Printf("[Autoscaler] Failed to spawn worker: %v", err)
				}
			}
		}
	}()
}