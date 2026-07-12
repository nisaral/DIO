package scheduler

import (
	"errors"
	"hash/fnv"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/nisaral/dio/api/proto"
	"github.com/nisaral/dio/internal/registry"
)

// PerWorkerPredictor implements Dual-Timescale NLMS (or single-µ / frozen static).
// Novelty: dual µ separates burst jitter (fast) from thermal/slow drift (slow).
type PerWorkerPredictor struct {
	mu               sync.Mutex
	FastSlope        float64
	SlowSlope        float64
	Intercept        float64
	AverageLatency   float64
	InterferenceHist [3]float64
	HistIdx          int
	SumSquaredError  float64
	SumAbsErr        float64
	SumRelErr        float64
	UpdateCount      int
	UseDualSlope     bool
	Frozen           bool // STATIC strategy: no online updates
	KVGrowthFactor   float64
	BandwidthPenalty float64
	Tier             string
	TotalVRAM        int64
	EngineType       string
	LastPredicted    float64
}

func (p *PerWorkerPredictor) effectiveSlopeLocked() float64 {
	if p.UseDualSlope {
		return (FastSlowBlend * p.FastSlope) + ((1 - FastSlowBlend) * p.SlowSlope)
	}
	// Single-µ: only fast slope (or only slope) drives prediction.
	return p.FastSlope
}

func (p *PerWorkerPredictor) modeName() string {
	if p.Frozen {
		return "STATIC"
	}
	if p.UseDualSlope {
		return "DUAL"
	}
	return "SINGLE"
}

func (p *PerWorkerPredictor) Update(actual float64, tokens int, vramMB int64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	bwPenalty := 1.0
	if vramMB < VRAMSoftLimitMB {
		bwPenalty = 1.0 + (VRAMSoftLimitMB-float64(vramMB))/VRAMSoftLimitMB
	}

	effectiveSlope := p.effectiveSlopeLocked()
	predicted := (effectiveSlope*float64(tokens) + p.Intercept) * bwPenalty
	p.LastPredicted = predicted
	err := actual - predicted

	absErr := math.Abs(err)
	relErr := absErr / math.Max(actual, 1.0)
	p.SumSquaredError += err * err
	p.SumAbsErr += absErr
	p.SumRelErr += relErr
	p.UpdateCount++
	if p.UpdateCount%25 == 0 {
		mse := p.SumSquaredError / float64(p.UpdateCount)
		mape := (p.SumRelErr / float64(p.UpdateCount)) * 100.0
		log.Printf("[NLMS_TELEMETRY] mode=%s MSE=%.4f MAPE=%.2f%% N=%d", p.modeName(), mse, mape, p.UpdateCount)
	}

	// Frozen static profile: track error but do not adapt (offline baseline).
	if p.Frozen {
		if p.AverageLatency == 0 {
			p.AverageLatency = actual
		} else {
			p.AverageLatency = 0.9*p.AverageLatency + 0.1*actual
		}
		return
	}

	if tokens > 0 {
		grad := err / float64(tokens)
		p.FastSlope += MuFast * grad
		if p.UseDualSlope {
			p.SlowSlope += MuSlow * grad
		} else {
			// Single-µ: keep slow slope glued to fast so dumps stay consistent.
			p.SlowSlope = p.FastSlope
		}
	}
	p.Intercept += MuBias * err

	if p.FastSlope < InitialSlope {
		p.FastSlope = InitialSlope
	}
	if p.SlowSlope < InitialSlope {
		p.SlowSlope = InitialSlope
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

	// Hard VRAM check also applied in cost.go; keep here for estimator-only paths.
	if vramMB > 0 && vramMB < VRAMHardLimitMB && tokens > 1000 {
		return math.MaxFloat64, p.AverageLatency
	}

	slope := p.effectiveSlopeLocked()
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
	if avgInterference < 1e-9 {
		avgInterference = 1.0
	}

	predicted := math.Max(base, memCost) * math.Max(1.0, avgInterference)
	p.LastPredicted = predicted
	return predicted, p.AverageLatency
}

func (p *PerWorkerPredictor) Snapshot() PredictorSnapshot {
	p.mu.Lock()
	defer p.mu.Unlock()
	mae, mape := 0.0, 0.0
	if p.UpdateCount > 0 {
		mae = p.SumAbsErr / float64(p.UpdateCount)
		mape = (p.SumRelErr / float64(p.UpdateCount)) * 100.0
	}
	return PredictorSnapshot{
		FastSlope:  p.FastSlope,
		SlowSlope:  p.SlowSlope,
		Intercept:  p.Intercept,
		AvgLatency: p.AverageLatency,
		Updates:    p.UpdateCount,
		Algorithm:  "NLMS",
		Mode:       p.modeName(),
		MAE:        mae,
		MAPE:       mape,
		Frozen:     p.Frozen,
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
	ablation      AblationFlags
	admission     AdmissionStats
	predHistory   *PredHistory
	lastMinScore  float64
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
	case "STATIC", "STATIC_PROFILE", "OFFLINE":
		return "STATIC"
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
	abl := LoadAblationFlags()
	nlmsMode := strings.ToUpper(strings.TrimSpace(os.Getenv("NLMS_MODE")))
	if abl.SingleTimescale {
		nlmsMode = "SINGLE"
	}
	if nlmsMode == "" {
		nlmsMode = "DUAL"
	}
	log.Printf("[SCHEDULER] strategy=%s nlms_mode=%s ablation=%s admission_off=%v slo_ms=%.0f",
		strategy, nlmsMode, abl.Name, AdmissionDisabled(), EffectiveSLOMs())
	return &Scheduler{
		workers:       make(map[string]*registry.Worker),
		predictors:    make(map[string]*PerWorkerPredictor),
		rlsPredictors: make(map[string]*PerWorkerRLSPredictor),
		statsHistory:  make(map[string][]DetailedMetrics),
		Strategy:      strategy,
		NLMSMode:      nlmsMode,
		prefixCache:   make(map[uint32]string),
		decisionLog:   make([]RoutingDecision, 0, 128),
		ablation:      abl,
		predHistory:   NewPredHistory(PredHistoryCap),
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

	useDual := s.NLMSMode != "SINGLE" && !s.ablation.SingleTimescale
	frozen := s.Strategy == "STATIC"
	slope := InitialSlope
	intercept := InitialIntercept
	if frozen {
		// Offline-calibrated baseline: env overrides for paper STATIC vs NLMS.
		if v := os.Getenv("STATIC_SLOPE"); v != "" {
			if f, err := strconvParse(v); err == nil {
				slope = f
			}
		} else {
			slope = StaticDefaultSlope
		}
		if v := os.Getenv("STATIC_INTERCEPT"); v != "" {
			if f, err := strconvParse(v); err == nil {
				intercept = f
			}
		} else {
			intercept = StaticDefaultIntercept
		}
	}
	s.predictors[id] = &PerWorkerPredictor{
		FastSlope:        slope,
		SlowSlope:        slope,
		Intercept:        intercept,
		UseDualSlope:     useDual && !frozen,
		Frozen:           frozen,
		KVGrowthFactor:   0.5,
		BandwidthPenalty: 1.5,
		Tier:             tier,
		TotalVRAM:        vramMB,
		EngineType:       engineType,
	}
	s.rlsPredictors[id] = NewRLSPredictor(tier, vramMB, engineType)
	log.Printf("[SCHEDULER] Worker %s registered. Tier: %s, VRAM: %d MB, Engine: %s, dual=%v frozen=%v",
		id, tier, vramMB, engineType, useDual && !frozen, frozen)
}

func strconvParse(v string) (float64, error) {
	return strconv.ParseFloat(v, 64)
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

	case "STATIC", "NLMS":
		fallthrough
	default:
		// STATIC uses frozen NLMS predictors; same cost path as NLMS.
		bestID, minScore, bestBreakdown, anyCandidate = s.pickPredictive(req, false)
		if bestID != "" {
			recordDecision(bestBreakdown)
		}
	}

	s.lastMinScore = minScore

	// Formal admission rule (goodput optimizer):
	//   reject if no feasible worker OR min_w S_w > SLO
	if bestID == "" {
		if !AdmissionDisabled() && anyCandidate && (minScore >= 1e300 || minScore > EffectiveSLOMs()) {
			retryMs := minScore
			if retryMs >= 1e300 {
				retryMs = 5000
				s.admission.RecordRejectVRAM()
			} else {
				s.admission.RecordRejectSLO()
			}
			return "", newAdmissionError(retryMs, "all workers exceed SLO or VRAM capacity")
		}
		s.admission.RecordRejectNoWorker()
		return "", errors.New("no healthy workers available")
	}

	// SLO admission: reject if predicted total wait+exec exceeds budget
	if !AdmissionDisabled() && (s.Strategy == "NLMS" || s.Strategy == "RLS" || s.Strategy == "STATIC") {
		if minScore > EffectiveSLOMs() {
			s.admission.RecordRejectSLO()
			return "", newAdmissionError(minScore, "predicted latency exceeds SLO")
		}
	}

	if w, ok := s.workers[bestID]; ok {
		w.PendingTasks++
		ph := prefixHash(req.Data)
		s.prefixCache[ph] = bestID
		s.admission.RecordAdmit()
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
			ablation:    s.ablation,
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
		var predicted float64
		var fast, slow float64
		mode := s.NLMSMode
		if pred, ok := s.predictors[workerID]; ok {
			// Pre-update prediction for honest MAPE (latency model only, ignore hard VRAM ∞).
			pred.mu.Lock()
			slope := pred.effectiveSlopeLocked()
			predicted = slope*float64(tokens) + pred.Intercept
			pred.mu.Unlock()
			pred.Update(metrics.E2ELatency, tokens, vram)
			snap := pred.Snapshot()
			fast, slow = snap.FastSlope, snap.SlowSlope
			mode = snap.Mode
		}
		if rp, ok := s.rlsPredictors[workerID]; ok {
			rp.Update(metrics.E2ELatency, tokens, vram)
		}
		s.statsHistory[workerID] = append(s.statsHistory[workerID], metrics)

		absErr := math.Abs(metrics.E2ELatency - predicted)
		rel := absErr / math.Max(metrics.E2ELatency, 1.0)
		if s.predHistory != nil && predicted > 0 && predicted < 1e300 {
			s.predHistory.Add(PredSample{
				UnixMs:    nowUnixMs(),
				WorkerID:  workerID,
				Tokens:    tokens,
				Predicted: predicted,
				Actual:    metrics.E2ELatency,
				AbsErr:    absErr,
				RelErr:    rel,
				FastSlope: fast,
				SlowSlope: slow,
				Mode:      mode,
			})
		}
		s.admission.RecordCompletion(metrics.E2ELatency, predicted)
	}

	if w, ok := s.workers[workerID]; ok && w.PendingTasks > 0 {
		w.PendingTasks--
	}
}

// GetAdmissionStats exports goodput/admission counters for the camera-ready suite.
func (s *Scheduler) GetAdmissionStats() map[string]interface{} {
	return s.admission.Snapshot()
}

// GetPredHistory exports dual-vs-single MAPE samples.
func (s *Scheduler) GetPredHistory(limit int) map[string]interface{} {
	if s.predHistory == nil {
		return map[string]interface{}{"count": 0, "samples": []PredSample{}}
	}
	return s.predHistory.Snapshot(limit)
}

// GetAblationInfo returns active ablation / mode configuration.
func (s *Scheduler) GetAblationInfo() map[string]interface{} {
	return map[string]interface{}{
		"name":               s.ablation.Name,
		"disable_queue":      s.ablation.DisableQueue,
		"disable_vram_soft":  s.ablation.DisableVRAMSoft,
		"disable_vram_hard":  s.ablation.DisableVRAMHard,
		"disable_tier":       s.ablation.DisableTier,
		"disable_cache":      s.ablation.DisableCache,
		"single_timescale":   s.ablation.SingleTimescale || s.NLMSMode == "SINGLE",
		"nlms_mode":          s.NLMSMode,
		"strategy":           s.Strategy,
		"slo_ms":             EffectiveSLOMs(),
		"admission_disabled": AdmissionDisabled(),
	}
}

// ResetAdmissionStats clears counters between suite experiments.
func (s *Scheduler) ResetAdmissionStats() {
	s.admission = AdmissionStats{}
	s.predHistory = NewPredHistory(PredHistoryCap)
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