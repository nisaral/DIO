package scheduler

import (
	"context"
	"log"
	"os"
	"sync"
	"time"

	"github.com/nisaral/dio/workers/worker_mgmt"
)

var (
	lastSpawnTime time.Time
	spawnMu       sync.Mutex
)

func StartAutoscaler(s *Scheduler, dm *worker_mgmt.DockerManager, threshold int) {
	if os.Getenv("AUTOSCALER_ENABLED") != "true" && os.Getenv("AUTOSCALER_ENABLED") != "1" {
		log.Println("[Autoscaler] Disabled (set AUTOSCALER_ENABLED=true to enable Docker spawn)")
		return
	}
	if dm == nil {
		log.Println("[Autoscaler] Docker manager unavailable; autoscaler disabled")
		return
	}

	network := os.Getenv("DOCKER_NETWORK")
	if network == "" {
		network = "dio_default"
	}
	managerAddr := os.Getenv("MANAGER_GRPC_ADDR")
	if managerAddr == "" {
		managerAddr = "dio-manager:50055"
	}

	ticker := time.NewTicker(15 * time.Second)
	go func() {
		for range ticker.C {
			workers := s.ListWorkers()
			workerCount := len(workers)
			totalQueue := s.GetGlobalQueueDepth()

			avgQueue := 0.0
			if workerCount > 0 {
				avgQueue = totalQueue / float64(workerCount)
			}

			if avgQueue > 5.0 || workerCount < threshold {
				log.Printf("[AUTOSCALER] Scale-up triggered. Avg Queue: %.2f, Workers: %d", avgQueue, workerCount)
				spawnMu.Lock()
				if time.Since(lastSpawnTime) < 30*time.Second {
					spawnMu.Unlock()
					continue
				}
				lastSpawnTime = time.Now()
				spawnMu.Unlock()

				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				err := dm.SpawnWorker(ctx, network, managerAddr)
				cancel()
				if err != nil {
					log.Printf("[AUTOSCALER] Spawn failed: %v", err)
				} else {
					log.Println("[AUTOSCALER] New worker container started")
				}
			} else if avgQueue < 1.0 && workerCount > threshold {
				log.Printf("[AUTOSCALER] Underutilized. Scale-down recommended. Avg Queue: %.2f", avgQueue)
			}
		}
	}()
}