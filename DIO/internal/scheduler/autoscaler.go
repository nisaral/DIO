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
			// Autoscaler Simplicity Fix:
			// Use Global Queue Depth / Worker Count ratio to determine saturation.
			// Target: Keep average queue depth < 5 per worker.

			// We access workers safely via Scheduler methods if possible,
			// but here we need a count. We can use ListWorkers.
			workers := s.ListWorkers()
			workerCount := len(workers)
			totalQueue := s.GetGlobalQueueDepth()

			avgQueue := 0.0
			if workerCount > 0 {
				avgQueue = totalQueue / float64(workerCount)
			}

			// Scale Up Condition:
			// 1. Average Queue > 5 (Saturation)
			// 2. OR Worker Count < Min Threshold (Cold Start)
			if avgQueue > 5.0 || workerCount < threshold {
				log.Println("[Autoscaler] Demand high. Spawning new Python worker...")
				ctx := context.Background()

				// Ensure you have a docker image named 'dio-worker' built
				err := dm.SpawnWorker(ctx, "dio-worker")
				if err != nil {
					log.Printf("[Autoscaler] Failed to spawn worker: %v", err)
				}
			}
		}
	}()
}
