package scheduler

import (
	"errors"
	"hash/fnv"
	"log"
	"math"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/nisaral/dio/api/proto"
	"github.com/nisaral/dio/internal/registry"
)

// PerWorkerPredictor implements the Dual-Timescale NLMS algorithm.
type PerWorkerPredictor struct {
	mu               sync.Mutex
	FastSlope        float64
	SlowSlope        float64
	Intercept        float64
	AverageLatency   float64
	InterferenceHist [3]float64
	HistIdx          int
	SumSquaredError  float64
	UpdateCount      int
	UseDualSlope     bool
	KVGrowthFactor   float64
	BandwidthPenalty float64
	Tier             string
	TotalVRAM        int64
	EngineType       string
}

func (p *PerWorkerPredictor) Update(actual float64, tokens int, vramMB int64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	bwPenalty := 1.0
	if vramMB < VRAMSoftLimitMB {
		bwPenalty = 1.0 + (VRAMSoftLimitMB-float64(vramMB))/VRAMSoftLimitMB
	}

	effectiveSlope := (FastSlowBlend * p.FastSlope) + ((1 - FastSlowBlend) * p.SlowSlope)
	predicted := (effectiveSlope*float64(tokens) + p.Intercept) * bwPenalty
	err := actual - predicted

	p.SumSquaredError += err * err
	p.UpdateCount++
	if p.UpdateCount%10 == 0 {
		mse := p.SumSquaredError / float64(p.UpdateCount)
		log.Printf("[NLMS_TELEMETRY] Worker MSE: %.4f (N=%d)", mse, p.UpdateCount)
	}

	if tokens > 0 {
		grad := err / float64(tokens)
		p.FastSlope += MuFast * grad
		if p.UseDualSlope {
			p.SlowSlope += MuSlow * grad
		}
	}
	p.Intercept += MuBias * err

	if p.FastSlope < InitialSlope {
		p.FastSlope = InitialSlope
	}
	if p.Intercept < 0.1 {
		p.Intercept = 0.1
	}

	ratio := actual / (predicted + 1e-9)
	p.InterferenceHist[p.HistIdx] = ratio
	p.HistIdx = (p.HistIdx + 1) % 3

	if p.AverageLatency == 0 {
		p.AverageLatency = actual
	} else {
		p.AverageLatency = (0.9 * p.AverageLatency) + (0.1 * actual)
	}
}

func (p *PerWorkerPredictor) Estimate(tokens int, vramMB int64) (float64, float64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if vramMB < VRAMHardLimitMB && tokens > 1000 {
		return math.MaxFloat64, p.AverageLatency
	}

	slope := (FastSlowBlend * p.FastSlope) + ((1 - FastSlowBlend) * p.SlowSlope)
	base := slope*float64(tokens) + p.Intercept

	memCost := 0.0
	if vramMB < VRAMSoftLimitMB {
		contention := (VRAMSoftLimitMB - float64(vramMB)) / VRAMSoftLimitMB
		memCost = base * contention * p.BandwidthPenalty
	}

	avgInterference := 0.0
	for _, v := range p.InterferenceHist {
		avgInterference += v
	}
	avgInterference /= 3.0

	predicted := math.Max(base, memCost)*math.Max(1.0, avgInterference)
	return predicted, p.AverageLatency
}

func (p *PerWorkerPredictor) Snapshot() PredictorSnapshot {
	p.mu.Lock()
	defer p.mu.Unlock()
	return PredictorSnapshot{
		FastSlope:  p.FastSlope,
		SlowSlope:  p.SlowSlope,
		Intercept:  p.Intercept,
		AvgLatency: p.AverageLatency,
		Updates:    p.UpdateCount,
		Algorithm:  "NLMS",
	}
}

type DetailedMetrics struct {
	TTFT             float64
	TPOT             float64
	E2ELatency       float64
	InputThroughput  float64
	OutputThroughput float64
}

type Scheduler struct {
	mu            sync.Mutex
	workers       map[string]*registry.Worker
	predictors    map[string]*PerWorkerPredictor
	rlsPredictors map[string]*PerWorkerRLSPredictor
	statsHistory  map[string][]DetailedMetrics
	Strategy      string
	rrIndex       uint64
	NLMSMode      string
	prefixCache   map[uint32]string
	lastDecision  *RoutingDecision
	decisionLog   []RoutingDecision
}

func normalizeStrategy(raw string) string {
	s := strings.TrimSpace(raw)
	if s == "" {
		return "NLMS"
	}
	upper := strings.ToUpper(s)
	switch upper {
	case "NLMS":
		return "NLMS"
	case "RLS":
		return "RLS"
	case "ROUNDROBIN", "ROUND_ROBIN":
		return "RoundRobin"
	case "LEASTLOADED", "LEAST_LOAD", "LEASTLOAD":
		return "LeastLoaded"
	case "LATENCYBASED", "LATENCY_BASED":
		return "LatencyBased"
	case "SESSION":
		return "Session"
	default:
		return s
	}
}

func NewScheduler() *Scheduler {
	strategy := normalizeStrategy(os.Getenv("SCHEDULER_STRATEGY"))
	nlmsMode := os.Getenv("NLMS_MODE")
	return &Scheduler{
		workers:       make(map[string]*registry.Worker),
		predictors:    make(map[string]*PerWorkerPredictor),
		rlsPredictors: make(map[string]*PerWorkerRLSPredictor),
		statsHistory:  make(map[string][]DetailedMetrics),
		Strategy:      strategy,
		NLMSMode:      nlmsMode,
		prefixCache:   make(map[uint32]string),
		decisionLog:   make([]RoutingDecision, 0, 128),
	}
}

func (s *Scheduler) RegisterWorker(id, address, tier string, vramMB int64) {
	s.RegisterWorkerWithEngine(id, address, tier, vramMB, "")
}

func (s *Scheduler) RegisterWorkerWithEngine(id, address, tier string, vramMB int64, engineType string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if vramMB <= 0 {
		vramMB = 24576
	}
	if engineType == "" {
		engineType = "hf"
	}

	s.workers[id] = &registry.Worker{
		ID:            id,
		Address:       address,
		IsHealthy:     true,
		LastKnownVRAM: vramMB,
		EngineType:    engineType,
	}

	s.predictors[id] = &PerWorkerPredictor{
		FastSlope:        InitialSlope,
		SlowSlope:        InitialSlope,
		Intercept:        InitialIntercept,
		UseDualSlope:     s.NLMSMode != "SINGLE",
		KVGrowthFactor:   0.5,
		BandwidthPenalty: 1.5,
		Tier:             tier,
		TotalVRAM:        vramMB,
		EngineType:       engineType,
	}
	s.rlsPredictors[id] = NewRLSPredictor(tier, vramMB, engineType)
	log.Printf("[SCHEDULER] Worker %s registered. Tier: %s, VRAM: %d MB, Engine: %s", id, tier, vramMB, engineType)
}

func (s *Scheduler) GetWorker(id string) (*registry.Worker, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	w, ok := s.workers[id]
	return w, ok
}

func (s *Scheduler) GetGlobalQueueDepth() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	var total float64
	for _, w := range s.workers {
		total += float64(w.PendingTasks)
	}
	return total
}

func (s *Scheduler) PickBestWorker(req *pb.InferenceRequest) (string, error) {
	start := time.Now()
	defer func() {
		overhead := time.Since(start).Nanoseconds()
		if overhead > 1000 {
			log.Printf("[SCHED_OVERHEAD] %d ns", overhead)
		}
	}()

	s.mu.Lock()
	defer s.mu.Unlock()

	var bestID string
	minScore := math.MaxFloat64
	var bestBreakdown RoutingDecision
	anyCandidate := false

	recordDecision := func(b RoutingDecision) {
		b.Strategy = s.Strategy
		s.lastDecision = &b
		s.decisionLog = append(s.decisionLog, b)
		if len(s.decisionLog) > 200 {
			s.decisionLog = s.decisionLog[len(s.decisionLog)-200:]
		}
	}

	switch s.Strategy {
	case "RoundRobin":
		workers := healthyWorkerIDs(s.workers)
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
			if w.PendingTasks < minTasks {
				minTasks = w.PendingTasks
				bestID = id
			}
		}

	case "LatencyBased":
		minLat := math.MaxFloat64
		for id, w := range s.workers {
			if !w.IsHealthy {
				continue
			}
			if pred, ok := s.predictors[id]; ok {
				lat := pred.AverageLatency
				if lat == 0 {
					lat = 0.1
				}
				if lat < minLat {
					minLat = lat
					bestID = id
				}
			}
		}

	case "Session":
		h := fnv.New32a()
		h.Write(req.Data)
		hash := h.Sum32()
		workers := healthyWorkerIDs(s.workers)
		if len(workers) > 0 {
			bestID = workers[hash%uint32(len(workers))]
		}

	case "RLS":
		bestID, minScore, bestBreakdown, anyCandidate = s.pickPredictive(req, true)
		if bestID != "" {
			recordDecision(bestBreakdown)
		}

	case "NLMS":
		fallthrough
	default:
		bestID, minScore, bestBreakdown, anyCandidate = s.pickPredictive(req, false)
		if bestID != "" {
			recordDecision(bestBreakdown)
		}
	}

	if bestID == "" {
		if anyCandidate && (minScore >= 1e300 || minScore > SLOTTFTMs) {
			retryMs := minScore
			if retryMs >= 1e300 {
				retryMs = 5000
			}
			return "", newAdmissionError(retryMs, "all workers exceed SLO or VRAM capacity")
		}
		return "", errors.New("no healthy workers available")
	}

	// SLO admission: reject if predicted total wait+exec exceeds budget
	if s.Strategy == "NLMS" || s.Strategy == "RLS" {
		if minScore > SLOTTFTMs {
			return "", newAdmissionError(minScore, "predicted latency exceeds SLO")
		}
	}

	if w, ok := s.workers[bestID]; ok {
		w.PendingTasks++
		ph := prefixHash(req.Data)
		s.prefixCache[ph] = bestID
		return bestID, nil
	}

	return bestID, nil
}

func (s *Scheduler) pickPredictive(req *pb.InferenceRequest, useRLS bool) (bestID string, minScore float64, bestBreakdown RoutingDecision, anyCandidate bool) {
	minScore = math.MaxFloat64

	for id, w := range s.workers {
		if !w.IsHealthy {
			continue
		}

		var estimator LatencyEstimator
		var tier string
		var totalVRAM int64

		if useRLS {
			rp, ok := s.rlsPredictors[id]
			if !ok {
				continue
			}
			estimator = rp
			tier = rp.Tier
			totalVRAM = rp.TotalVRAM
		} else {
			np, ok := s.predictors[id]
			if !ok {
				continue
			}
			estimator = np
			tier = np.Tier
			totalVRAM = np.TotalVRAM
		}

		score, breakdown, blocked := computeWorkerScore(workerScoreInput{
			req:         req,
			worker:      w,
			estimator:   estimator,
			tier:        tier,
			totalVRAM:   totalVRAM,
			prefixCache: s.prefixCache,
		})
		if blocked {
			continue
		}
		anyCandidate = true
		if score < minScore {
			minScore = score
			bestID = id
			bestBreakdown = breakdown
		}
	}
	return bestID, minScore, bestBreakdown, anyCandidate
}

func healthyWorkerIDs(workers map[string]*registry.Worker) []string {
	out := make([]string, 0, len(workers))
	for id, w := range workers {
		if w.IsHealthy {
			out = append(out, id)
		}
	}
	return out
}

func (s *Scheduler) FeedbackLoop(workerID string, metrics DetailedMetrics, tokens int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	vram := int64(24000)
	if w, ok := s.workers[workerID]; ok {
		vram = w.LastKnownVRAM
	}

	if metrics.E2ELatency > 0 {
		if pred, ok := s.predictors[workerID]; ok {
			pred.Update(metrics.E2ELatency, tokens, vram)
		}
		if rp, ok := s.rlsPredictors[workerID]; ok {
			rp.Update(metrics.E2ELatency, tokens, vram)
		}
		s.statsHistory[workerID] = append(s.statsHistory[workerID], metrics)
	}

	if w, ok := s.workers[workerID]; ok && w.PendingTasks > 0 {
		w.PendingTasks--
	}
}

func (s *Scheduler) CancelRequest(workerID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if w, ok := s.workers[workerID]; ok && w.PendingTasks > 0 {
		w.PendingTasks--
		log.Printf("[SCHEDULER] Client disconnected. Cancelling task on worker %s", workerID)
	}
}

func (s *Scheduler) ResetWorkerState(workerID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if pred, ok := s.predictors[workerID]; ok {
		pred.mu.Lock()
		pred.FastSlope = InitialSlope
		pred.SlowSlope = InitialSlope
		pred.Intercept = InitialIntercept
		pred.mu.Unlock()
	}
	if rp, ok := s.rlsPredictors[workerID]; ok {
		rp.mu.Lock()
		rp.Slope = InitialSlope
		rp.Intercept = InitialIntercept
		rp.P = [2][2]float64{{1000, 0}, {0, 1000}}
		rp.mu.Unlock()
	}
}

func (s *Scheduler) GetDebugPrediction(workerID string, tokens int) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if pred, ok := s.predictors[workerID]; ok {
		est, _ := pred.Estimate(tokens, 24000)
		return est
	}
	return 0.0
}

func (s *Scheduler) ListWorkers() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := make([]string, 0, len(s.workers))
	for id := range s.workers {
		ids = append(ids, id)
	}
	return ids
}

func (s *Scheduler) GetWorkerMetrics() []map[string]interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]map[string]interface{}, 0, len(s.workers))
	for id, w := range s.workers {
		entry := map[string]interface{}{
			"id":            id,
			"address":       w.Address,
			"healthy":       w.IsHealthy,
			"pending":       w.PendingTasks,
			"free_vram_mb":  w.LastKnownVRAM,
			"engine":        w.EngineType,
		}
		if pred, ok := s.predictors[id]; ok {
			snap := pred.Snapshot()
			entry["nlms"] = snap
			entry["tier"] = pred.Tier
		}
		if rp, ok := s.rlsPredictors[id]; ok {
			entry["rls"] = rp.Snapshot()
		}
		out = append(out, entry)
	}
	return out
}

func (s *Scheduler) GetLastDecision() *RoutingDecision {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.lastDecision == nil {
		return nil
	}
	d := *s.lastDecision
	return &d
}

func (s *Scheduler) GetDecisionLog() []RoutingDecision {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]RoutingDecision, len(s.decisionLog))
	copy(out, s.decisionLog)
	return out
}

func (s *Scheduler) SetWorkerVRAM(workerID string, freeMB int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if w, ok := s.workers[workerID]; ok {
		w.LastKnownVRAM = freeMB
	}
}

func (s *Scheduler) SetWorkerLatencyMultiplier(workerID string, mult float64) {
	// Reserved for chaos injection via demo dashboard.
	_ = workerID
	_ = mult
}