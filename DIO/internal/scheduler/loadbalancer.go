package scheduler

import (
	"errors"
	"log"
	"math"
	"sync"
	"time"

	pb "github.com/nisaral/dio/api/proto"
	"github.com/nisaral/dio/internal/registry"
)

// PerWorkerPredictor implements the Dual-Timescale RLS algorithm
type PerWorkerPredictor struct {
	mu               sync.Mutex
	FastSlope        float64 // Fast adaptation (alpha=0.1)
	SlowSlope        float64 // Slow adaptation (alpha=0.01) for drift
	Intercept        float64
	AverageLatency   float64
	InterferenceHist [3]float64 // Ring buffer for co-location noise
	HistIdx          int
	// Telemetry: RLS Convergence
	SumSquaredError float64
	UpdateCount     int
}

// Update implements the Recursive Least Squares (RLS) update step.
// Mathematical Derivation (Sherman-Morrison):
// We approximate the covariance matrix update P(t) = P(t-1) - ...
// by using a dual-timescale gradient descent which is computationally O(1).
// Cost Function J(theta) = sum(lambda^(t-i) * (y_i - theta^T x_i)^2)
func (p *PerWorkerPredictor) Update(actual float64, tokens int, vram int64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// 1. Calculate Error based on current model
	// Roofline Penalty: If VRAM < 4GB, assume bandwidth contention
	bwPenalty := 1.0
	if vram < 4096 {
		bwPenalty = 1.0 + (4096.0-float64(vram))/4096.0
	}

	// Weighted slope (favor fast reaction for spikes)
	effectiveSlope := (0.8 * p.FastSlope) + (0.2 * p.SlowSlope)
	predicted := (effectiveSlope*float64(tokens) + p.Intercept) * bwPenalty

	error := actual - predicted

	// Telemetry: Log MSE for Learning Curve
	p.SumSquaredError += error * error
	p.UpdateCount++
	if p.UpdateCount%10 == 0 {
		mse := p.SumSquaredError / float64(p.UpdateCount)
		log.Printf("[RLS_TELEMETRY] Worker MSE: %.4f (N=%d)", mse, p.UpdateCount)
	}

	// 2. Dual-Timescale Update (The "Research Novelty")
	if tokens > 0 {
		grad := error / float64(tokens)
		p.FastSlope += 0.1 * grad  // Fast learning rate
		p.SlowSlope += 0.01 * grad // Slow learning rate
	}
	p.Intercept += 0.01 * error

	// Constraints
	if p.FastSlope < 0.1 {
		p.FastSlope = 0.1
	}

	// 3. Update Interference History
	ratio := actual / (predicted + 1e-9)
	p.InterferenceHist[p.HistIdx] = ratio
	p.HistIdx = (p.HistIdx + 1) % 3

	// 4. Update Average Latency (EMA with alpha=0.1)
	if p.AverageLatency == 0 {
		p.AverageLatency = actual
	} else {
		p.AverageLatency = (0.9 * p.AverageLatency) + (0.1 * actual)
	}
}

// Estimate function for the Scheduler to call
func (p *PerWorkerPredictor) Estimate(tokens int, vram int64) (float64, float64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	slope := (0.8 * p.FastSlope) + (0.2 * p.SlowSlope)
	base := slope*float64(tokens) + p.Intercept

	// Apply Interference Multiplier from history
	avgInterference := 0.0
	for _, v := range p.InterferenceHist {
		avgInterference += v
	}
	avgInterference /= 3.0

	predicted := base * math.Max(1.0, avgInterference)
	return predicted, p.AverageLatency
}

type DetailedMetrics struct {
	TTFT             float64
	TPOT             float64
	E2ELatency       float64
	InputThroughput  float64
	OutputThroughput float64
}

type Scheduler struct {
	mu           sync.Mutex
	workers      map[string]*registry.Worker
	predictors   map[string]*PerWorkerPredictor
	statsHistory map[string][]DetailedMetrics
	Strategy     string // "RLS" (Default), "LeastLoaded" (Baseline)
}

func NewScheduler() *Scheduler {
	return &Scheduler{
		workers:      make(map[string]*registry.Worker),
		predictors:   make(map[string]*PerWorkerPredictor),
		statsHistory: make(map[string][]DetailedMetrics),
		Strategy:     "RLS",
	}
}

func (s *Scheduler) RegisterWorker(id, address string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.workers[id] = &registry.Worker{
		ID:            id,
		Address:       address,
		IsHealthy:     true,
		LastKnownVRAM: 24000, // Initialize to 24GB to avoid initial roofline penalty
	}
	// Init predictor for new worker
	s.predictors[id] = &PerWorkerPredictor{
		FastSlope: 0.1,
		SlowSlope: 0.1,
		Intercept: 50.0,
	}
}

// GetWorker safely retrieves a worker by ID
func (s *Scheduler) GetWorker(id string) (*registry.Worker, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	w, ok := s.workers[id]
	return w, ok
}

// PickBestWorker implements SJF (Shortest Job First)
func (s *Scheduler) PickBestWorker(req *pb.InferenceRequest) (string, error) {
	start := time.Now()
	defer func() {
		// Telemetry: Scheduling Overhead (O(1) Proof)
		overhead := time.Since(start).Nanoseconds()
		if overhead > 1000 { // Log only significant overheads (>1us) to reduce noise
			log.Printf("[SCHED_OVERHEAD] %d ns", overhead)
		}
	}()

	s.mu.Lock()
	defer s.mu.Unlock()

	var bestID string
	minScore := math.MaxFloat64

	// Baseline: Least Loaded (NexusSched Simulation)
	if s.Strategy == "LeastLoaded" {
		minTasks := math.MaxInt
		for id, w := range s.workers {
			if !w.IsHealthy {
				continue
			}
			if w.PendingTasks < minTasks {
				minTasks = w.PendingTasks
				bestID = id
			}
		}
		if bestID == "" {
			return "", errors.New("no healthy workers available")
		}
		s.workers[bestID].PendingTasks++
		return bestID, nil
	}

	// Approximate tokens (chars / 4)
	tokens := len(req.Data) / 4

	for id, w := range s.workers {
		if !w.IsHealthy {
			continue
		}

		pred, ok := s.predictors[id]
		if !ok {
			continue
		}

		// 1. Execution Cost (RLS Prediction)
		// NEW: Use the Research Predictor
		execTime, avgLatency := pred.Estimate(tokens, w.LastKnownVRAM)

		// 2. Wait Cost (Queue Depth)
		// Little's Law: Wait = QueueLength * AverageServiceTime
		avgTaskTime := avgLatency
		if avgTaskTime == 0 {
			avgTaskTime = execTime // Fallback if no history
		}
		waitTime := float64(w.PendingTasks) * avgTaskTime

		// Total Cost = Wait + Exec
		score := waitTime + execTime

		if score < minScore {
			minScore = score
			bestID = id
		}
	}

	if bestID == "" {
		return "", errors.New("no healthy workers available")
	}

	// Increment queue depth (Optimistic concurrency)
	s.workers[bestID].PendingTasks++
	return bestID, nil
}

// FeedbackLoop updates RLS and decrements queue
func (s *Scheduler) FeedbackLoop(workerID string, metrics DetailedMetrics, tokens int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Update Predictor
	if pred, ok := s.predictors[workerID]; ok {
		vram := int64(24000) // Default
		if w, ok := s.workers[workerID]; ok {
			vram = w.LastKnownVRAM
		}
		if metrics.E2ELatency > 0 {
			pred.Update(metrics.E2ELatency, tokens, vram)
			// Store detailed metrics for rigorous analysis
			s.statsHistory[workerID] = append(s.statsHistory[workerID], metrics)
		}
	}

	// Decrement Queue
	if w, ok := s.workers[workerID]; ok {
		if w.PendingTasks > 0 {
			w.PendingTasks--
		}
	}
}
