package scheduler

import (
	"log"
	"time"

	"github.com/nisaral/dio/workers/worker_mgmt"
)

func StartAutoscaler(s *Scheduler, dm *worker_mgmt.DockerManager, threshold int) {
	ticker := time.NewTicker(15 * time.Second)
	
	// Track scaling state to avoid log spamming
	go func() {
		for range ticker.C {
			// 1. Safe access to scheduler state
			workers := s.ListWorkers()
			workerCount := len(workers)
			totalQueue := s.GetGlobalQueueDepth()

			avgQueue := 0.0
			if workerCount > 0 {
				avgQueue = totalQueue / float64(workerCount)
			}

			// 2. Determine Scaling Condition
			// Research Logic: Scale if queue > 5 OR we haven't reached the minimum experiment threshold
			if avgQueue > 5.0 || workerCount < threshold {
				log.Printf("[AUTOSCALER_ANALYTICS] Scale-up triggered. Avg Queue: %.2f, Current Workers: %d", avgQueue, workerCount)
				
				// NOTE FOR RESEARCH:
				// In a production Kubernetes/Docker environment, we call dm.SpawnWorker here.
				// In Lightning AI Bare-Metal, we manually launch workers with CUDA_VISIBLE_DEVICES 
				// to ensure precise hardware mapping (A100 vs T4 isolation).
				
				if dm != nil {
					log.Println("[Autoscaler] Manual intervention required: Please spawn a new worker process in a new terminal.")
				}
			} else if avgQueue < 1.0 && workerCount > threshold {
				log.Printf("[AUTOSCALER_ANALYTICS] Cluster underutilized. Scale-down recommended. Avg Queue: %.2f", avgQueue)
			}
		}
	}()
}