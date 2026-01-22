package scheduler

import (
	"sync"

	pb "github.com/nisaral/dio/api/proto"
	"github.com/nisaral/dio/internal/registry"
)

type PerWorkerPredictor struct {
	Slope          float64
	Intercept      float64
	AverageLatency float64
}

func (p *PerWorkerPredictor) Update(actual float64, tokens int, vram int64) {
	// Simple Online Learning (LMS)
	// Predicted = Slope * Tokens + Intercept
	predicted := p.Slope*float64(tokens) + p.Intercept
	error := actual - predicted
	lr := 0.001 // Learning Rate

	// Update weights
	p.Slope += lr * error * float64(tokens)
	p.Intercept += lr * error

	// Constraints to prevent instability
	if p.Slope < 0.1 {
		p.Slope = 0.1
	}
	if p.Intercept < 0 {
		p.Intercept = 0
	}
}

type Scheduler struct {
	mu         sync.Mutex
	workers    map[string]*registry.Worker
	predictors map[string]*PerWorkerPredictor
}

func NewScheduler() *Scheduler {
	return &Scheduler{
		workers:    make(map[string]*registry.Worker),
		predictors: make(map[string]*PerWorkerPredictor),
	}
}

// UpdateWorkers refreshes the list of available workers from the registry
func (s *Scheduler) UpdateWorkers(workers []*registry.Worker) {
	s.mu.Lock()
	defer s.mu.Unlock()

	newMap := make(map[string]*registry.Worker)
	for _, w := range workers {
		// Preserve existing state (Channel) if worker already exists
		if existing, ok := s.workers[w.ID]; ok && existing.TaskChannel != nil {
			w.TaskChannel = existing.TaskChannel
		} else if w.TaskChannel == nil {
			// Initialize channel if new or nil
			w.TaskChannel = make(chan *pb.InferenceRequest, 10)
		}

		newMap[w.ID] = w
		// Initialize predictor if not exists
		if _, exists := s.predictors[w.ID]; !exists {
			s.predictors[w.ID] = &PerWorkerPredictor{
				Slope:          2.1, // Initial heuristic
				Intercept:      50,
				AverageLatency: 100,
			}
		}
	}
	s.workers = newMap
}

// FeedbackLoop allows the scheduler to learn from real-world performance
func (s *Scheduler) FeedbackLoop(workerID string, actualLatency float64, tokens int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	pred, exists := s.predictors[workerID]
	if !exists {
		return
	}

	// Pass the actual VRAM from the worker's latest heartbeat for Roofline accuracy
	worker, exists := s.workers[workerID]
	if !exists {
		return
	}

	// Online Update: Nudges the Slope and Intercept
	pred.Update(actualLatency, tokens, worker.LastKnownVRAM)

	// Update moving average for wait-time estimation
	pred.AverageLatency = (pred.AverageLatency * 0.9) + (actualLatency * 0.1)
}

func (s *Scheduler) calculateWaitTime(w *registry.Worker, pred *PerWorkerPredictor) float64 {
	// Current Queue Depth * Learned Average Latency
	return float64(len(w.TaskChannel)) * pred.AverageLatency
}

// PickBestWorker selects the worker with the lowest predicted Total Completion Time (SJF)
func (s *Scheduler) PickBestWorker(tokens int) *registry.Worker {
	s.mu.Lock()
	defer s.mu.Unlock()

	var bestWorker *registry.Worker
	minCost := 1e12 // Large number

	for _, w := range s.workers {
		if !w.IsHealthy {
			continue
		}

		pred, ok := s.predictors[w.ID]
		if !ok {
			continue
		}

		// Cost = Execution Time + Wait Time
		execTime := pred.Slope*float64(tokens) + pred.Intercept
		waitTime := s.calculateWaitTime(w, pred)
		totalCost := execTime + waitTime

		if totalCost < minCost {
			minCost = totalCost
			bestWorker = w
		}
	}

	return bestWorker
}
