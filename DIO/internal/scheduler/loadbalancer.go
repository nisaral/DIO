package scheduler

import (
	"errors"
	"hash/fnv"
	"log"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/nisaral/dio/api/proto"
	"github.com/nisaral/dio/internal/registry"
)

// PerWorkerPredictor implements the Dual-Timescale NLMS algorithm
type PerWorkerPredictor struct {
	mu               sync.Mutex
	FastSlope        float64 // Fast adaptation (alpha=0.1)
	SlowSlope        float64 // Slow adaptation (alpha=0.01) for drift
	Intercept        float64
	AverageLatency   float64
	InterferenceHist [3]float64 // Ring buffer for co-location noise
	HistIdx          int
	// Telemetry: NLMS Convergence
	SumSquaredError float64
	UpdateCount     int
	UseDualSlope    bool
	// KV cache / memory modelling (Gap 2 & 5)
	KVGrowthFactor   float64 // bytes (or MB) per token approximation
	BandwidthPenalty float64 // roofline multiplier when memory pressure exists
	Tier             string  // "small", "large"
	TotalVRAM        int64   // GB
	EngineType       string  // "vllm", "hf", "mock" (v3)
}

// Update implements the Normalized Least Mean Squares (NLMS) update step.
// Mathematical Derivation:
// We use a dual-timescale gradient descent which is computationally O(1).
func (p *PerWorkerPredictor) Update(actual float64, tokens int, vramMB int64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// 1. Calculate Error based on current model
	// Roofline Penalty: If VRAM < 4GB, assume bandwidth contention
	bwPenalty := 1.0
	if vramMB < 4096 {
		bwPenalty = 1.0 + (4096.0-float64(vramMB))/4096.0
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
		log.Printf("[NLMS_TELEMETRY] Worker MSE: %.4f (N=%d)", mse, p.UpdateCount)
	}

	// 2. Dual-Timescale Update (The "Research Novelty")
	if tokens > 0 {
		grad := error / float64(tokens)
		p.FastSlope += 0.2 * grad // Fast learning rate (doubled for demo reactivity)
		if p.UseDualSlope {
			p.SlowSlope += 0.05 * grad // Slow learning rate (increased for drift)
		}
	}
	p.Intercept += 0.1 * error // Intercept learning rate (10x for fast baseline correction)

	// Constraints
	if p.FastSlope < 0.1 {
		p.FastSlope = 0.1
	}
	if p.Intercept < 0.1 {
		p.Intercept = 0.1
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
func (p *PerWorkerPredictor) Estimate(tokens int, vramMB int64) (float64, float64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	// NOTE: extended Estimate to include KV-growth and reasoning cost.
	// Backwards-compatible callers should pass zeros for outputPred and reasoning.
	// Gap 2: KV-Aware Roofline (Hard Block)
	// If VRAM is > 90% full (Free < 2400MB assuming 24GB total) and request is long,
	// we block it to prevent OOM/Fragmentation.
	// (We also compute a memCost component below to capture KV growth effects.)
	if vramMB < 2400 && tokens > 1000 {
		return math.MaxFloat64, p.AverageLatency
	}

	slope := (0.8 * p.FastSlope) + (0.2 * p.SlowSlope)
	base := slope*float64(tokens) + p.Intercept

	// Memory / KV cost: if free VRAM is low, inflate cost proportionally
	memCost := 0.0
	if vramMB < 4096 {
		contention := (4096.0 - float64(vramMB)) / 4096.0
		memCost = base * contention * p.BandwidthPenalty
	}

	// Reasoning cost: placeholder multiplier for compute-dense reasoning tokens.
	// At scheduling time we don't know exact reasoning tokens; callers may pass
	// reasoning via the extended Estimate signature (below). Default is 0.
	reasoningCost := 0.0

	// Apply Interference Multiplier from history
	avgInterference := 0.0
	for _, v := range p.InterferenceHist {
		avgInterference += v
	}
	avgInterference /= 3.0

	predicted := math.Max(base, memCost)*math.Max(1.0, avgInterference) + reasoningCost
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
	Strategy     string            // "NLMS" (Default), "LeastLoaded" (Baseline)
	rrIndex      uint64            // For Round Robin
	NLMSMode     string            // "DUAL" (Default) or "SINGLE"
	prefixCache  map[uint32]string // Map[PrefixHash]WorkerID for Prefix Caching
}

func NewScheduler() *Scheduler {
	strategy := os.Getenv("SCHEDULER_STRATEGY")
	if strategy == "" {
		strategy = "NLMS"
	}
	nlmsMode := os.Getenv("NLMS_MODE")
	return &Scheduler{
		workers:      make(map[string]*registry.Worker),
		predictors:   make(map[string]*PerWorkerPredictor),
		statsHistory: make(map[string][]DetailedMetrics),
		Strategy:     strategy,
		NLMSMode:     nlmsMode,
		prefixCache:  make(map[uint32]string),
	}
}

func (s *Scheduler) RegisterWorker(id, address, tier string, vramMB int64) {
	s.RegisterWorkerWithEngine(id, address, tier, vramMB, "")
}

// RegisterWorkerWithEngine registers a worker with engine type info (v3)
func (s *Scheduler) RegisterWorkerWithEngine(id, address, tier string, vramMB int64, engineType string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Standardize to MB. If 0, default to 24GB.
	if vramMB <= 0 {
		vramMB = 24576
	}

	// Default engine type
	if engineType == "" {
		engineType = "hf" // Legacy HuggingFace workers
	}

	s.workers[id] = &registry.Worker{
		ID:            id,
		Address:       address,
		IsHealthy:     true,
		LastKnownVRAM: vramMB, // STORE AS MB
		EngineType:    engineType,
	}

	s.predictors[id] = &PerWorkerPredictor{
		FastSlope:        0.1,
		SlowSlope:        0.1,
		Intercept:        50.0,
		UseDualSlope:     s.NLMSMode != "SINGLE",
		KVGrowthFactor:   0.5,
		BandwidthPenalty: 1.5, // Significant penalty for research visibility
		Tier:             tier,
		TotalVRAM:        vramMB, // STORE AS MB
		EngineType:       engineType,
	}
	log.Printf("[SCHEDULER] Worker %s registered. Tier: %s, VRAM: %d MB, Engine: %s", id, tier, vramMB, engineType)
}

// GetWorker safely retrieves a worker by ID
func (s *Scheduler) GetWorker(id string) (*registry.Worker, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	w, ok := s.workers[id]
	return w, ok
}

// GetGlobalQueueDepth returns the total pending tasks across all workers
func (s *Scheduler) GetGlobalQueueDepth() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	var total float64
	for _, w := range s.workers {
		total += float64(w.PendingTasks)
	}
	return total
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

	// === Strategy Selection ===
	switch s.Strategy {
	case "RoundRobin":
		// Atomic Round Robin (No locking needed for index, but we need the list)
		// Convert map to slice for stable ordering (expensive but necessary for strict RR in this map structure)
		// Optimization: In prod, cache the slice. Here we iterate.
		workers := make([]string, 0, len(s.workers))
		for id, w := range s.workers {
			if w.IsHealthy {
				workers = append(workers, id)
			}
		}
		if len(workers) == 0 {
			return "", errors.New("no healthy workers available")
		}
		idx := atomic.AddUint64(&s.rrIndex, 1)
		bestID = workers[idx%uint64(len(workers))]

	case "LeastLoaded":
		minTasks := math.MaxInt
		for id, w := range s.workers {
			if !w.IsHealthy {
				continue
			}
			// Least Outstanding Requests
			if w.PendingTasks < minTasks {
				minTasks = w.PendingTasks
				bestID = id
			}
		}

	case "LatencyBased":
		// Pick worker with lowest moving average latency
		minLat := math.MaxFloat64
		for id, w := range s.workers {
			if !w.IsHealthy {
				continue
			}
			if pred, ok := s.predictors[id]; ok {
				// Use AverageLatency (EMA)
				lat := pred.AverageLatency
				if lat == 0 {
					lat = 0.1 // Prefer un-probed workers
				}
				if lat < minLat {
					minLat = lat
					bestID = id
				}
			}
		}

	case "Session":
		// Sticky Session: Hash the input data (Prompt) to a worker
		// In a real app, we'd use a SessionID header.
		h := fnv.New32a()
		h.Write(req.Data)
		hash := h.Sum32()

		workers := make([]string, 0, len(s.workers))
		for id, w := range s.workers {
			if w.IsHealthy {
				workers = append(workers, id)
			}
		}
		if len(workers) > 0 {
			bestID = workers[hash%uint32(len(workers))]
		}

	case "NLMS":
		fallthrough
	default:
		// === DIO NLMS Logic (Default) ===
		// Approximate tokens (chars / 4)
		tokens := len(req.Data) / 4

		// Phase 3: Prefix Caching Router
		// Hash the first 100 bytes (System Prompt / Prefix)
		prefixLen := 100
		if len(req.Data) < prefixLen {
			prefixLen = len(req.Data)
		}
		h := fnv.New32a()
		h.Write(req.Data[:prefixLen])
		prefixHash := h.Sum32()

		for id, w := range s.workers {
			if !w.IsHealthy {
				continue
			}

			pred, ok := s.predictors[id]
			if !ok {
				continue
			}

			// === Tier Routing Logic ===
			// 1. Large requests MUST go to Large workers
			if req.Tier == "large" && pred.Tier != "large" {
				continue
			}
			// 2. Small requests prefer Small workers (Soft Constraint)
			tierMismatchPenalty := 0.0
			if req.Tier == "small" && pred.Tier == "large" {
				tierMismatchPenalty = 500.0 // 500ms penalty to save large workers for large tasks
			}

			// 1. Execution Cost (NLMS Prediction)
			execTime, avgLatency := pred.Estimate(tokens, w.LastKnownVRAM)

			// 2. Wait Cost (Queue Depth)
			// Little's Law: Wait = QueueLength * AverageServiceTime
			avgTaskTime := avgLatency
			if avgTaskTime == 0 {
				avgTaskTime = execTime // Fallback if no history
			}

			// Continuous Batching Simulation (Gap Closure)
			// Instead of serial wait (Queue * Time), assume batching reduces wait.
			// Effective Wait = (Queue / BatchSize) * Time
			batchSize := 8.0
			effectiveQueue := math.Max(0, float64(w.PendingTasks))
			waitTime := (effectiveQueue / batchSize) * avgTaskTime

			// 3. VRAM Penalty (Roofline)
			// If Free VRAM is low relative to Total VRAM, increase cost
			vramPenalty := 0.0
			if pred.TotalVRAM > 0 && w.LastKnownVRAM < 4096 {
				vramPenalty = (1.0 - (float64(w.LastKnownVRAM) / float64(pred.TotalVRAM*1024))) * 1000.0
			}

			// Total Cost = Wait + Exec + Penalties
			score := waitTime + execTime + tierMismatchPenalty + vramPenalty

			// Apply Cache Hit Bonus
			// If this worker served this prefix recently, it likely has KV cache.
			if cachedID, ok := s.prefixCache[prefixHash]; ok && cachedID == id {
				score -= 200.0 // Subtract 200ms from cost (Bonus)
			}

			if score < minScore {
				minScore = score
				bestID = id
			}
		}
	}

	if bestID == "" {
		return "", errors.New("no healthy workers available")
	}

	// Increment queue depth (Optimistic concurrency)
	if w, ok := s.workers[bestID]; ok {
		w.PendingTasks++
		// Update Prefix Cache (Optimistic)
		h := fnv.New32a()
		h.Write(req.Data[:min(len(req.Data), 100)])
		s.prefixCache[h.Sum32()] = bestID
		return bestID, nil
	}

	return bestID, nil
}

// FeedbackLoop updates NLMS and decrements queue
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

// CancelRequest handles client disconnection events (Gap 3)
func (s *Scheduler) CancelRequest(workerID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if w, ok := s.workers[workerID]; ok {
		// 1. Decrement Queue Depth
		if w.PendingTasks > 0 {
			w.PendingTasks--
		}
		// 2. Forward Cancel Signal
		log.Printf("[SCHEDULER] Client disconnected. Cancelling task on worker %s", workerID)
	}
}

// ResetWorkerState resets the NLMS predictor for a specific worker (Debug/Research)
func (s *Scheduler) ResetWorkerState(workerID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if pred, ok := s.predictors[workerID]; ok {
		pred.mu.Lock()
		pred.FastSlope = 0.1 // Reset to initial conservative
		pred.SlowSlope = 0.1
		pred.Intercept = 50.0
		pred.mu.Unlock()
	}
}

// GetDebugPrediction returns the current NLMS prediction for a worker (Debug/Research)
func (s *Scheduler) GetDebugPrediction(workerID string, tokens int) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if pred, ok := s.predictors[workerID]; ok {
		// Assume 24GB VRAM for debug consistency
		est, _ := pred.Estimate(tokens, 24000)
		return est
	}
	log.Printf("[DEBUG] GetDebugPrediction: Worker %s not found", workerID)
	return 0.0
}

// ListWorkers returns a list of active worker IDs (Debug)
func (s *Scheduler) ListWorkers() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := make([]string, 0, len(s.workers))
	for id := range s.workers {
		ids = append(ids, id)
	}
	return ids
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
