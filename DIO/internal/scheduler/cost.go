package scheduler

import (
	"hash/fnv"

	pb "github.com/nisaral/dio/api/proto"
	"github.com/nisaral/dio/internal/registry"
)

// LatencyEstimator predicts execution time for a request on a worker.
type LatencyEstimator interface {
	Estimate(tokens int, vramMB int64) (execMs float64, avgLatencyMs float64)
}

// RoutingDecision captures the last cost breakdown for observability.
type RoutingDecision struct {
	WorkerID    string  `json:"worker_id"`
	ExecMs      float64 `json:"exec_ms"`
	WaitMs      float64 `json:"wait_ms"`
	TierCostMs  float64 `json:"tier_cost_ms"`
	VRAMCostMs  float64 `json:"vram_cost_ms"`
	CacheBonusMs float64 `json:"cache_bonus_ms"`
	TotalMs     float64 `json:"total_ms"`
	Tokens      int     `json:"tokens"`
	Strategy    string  `json:"strategy"`
}

// PredictorSnapshot exposes per-worker model state for the dashboard.
type PredictorSnapshot struct {
	FastSlope  float64 `json:"fast_slope"`
	SlowSlope  float64 `json:"slow_slope"`
	Intercept  float64 `json:"intercept"`
	AvgLatency float64 `json:"avg_latency_ms"`
	Updates    int     `json:"updates"`
	Algorithm  string  `json:"algorithm"`
}

type workerScoreInput struct {
	req         *pb.InferenceRequest
	worker      *registry.Worker
	estimator   LatencyEstimator
	tier        string
	totalVRAM   int64
	prefixCache map[uint32]string
}

func prefixHash(data []byte) uint32 {
	prefixLen := 100
	if len(data) < prefixLen {
		prefixLen = len(data)
	}
	if prefixLen == 0 {
		return 0
	}
	h := fnv.New32a()
	h.Write(data[:prefixLen])
	return h.Sum32()
}

func computeWorkerScore(in workerScoreInput) (score float64, breakdown RoutingDecision, blocked bool) {
	tokens := len(in.req.Data) / 4
	if tokens < 1 {
		tokens = 1
	}

	execTime, avgLatency := in.estimator.Estimate(tokens, in.worker.LastKnownVRAM)
	if execTime >= 1e300 {
		return 0, RoutingDecision{}, true
	}

	avgTaskTime := avgLatency
	if avgTaskTime == 0 {
		avgTaskTime = execTime
	}

	waitTime := (float64(in.worker.PendingTasks) / BatchSize) * avgTaskTime

	tierPenalty := 0.0
	if in.req.Tier == "large" && in.tier != "large" {
		return 0, RoutingDecision{}, true
	}
	if in.req.Tier == "small" && in.tier == "large" {
		tierPenalty = TierMismatchMs
	}

	vramPenalty := 0.0
	if in.totalVRAM > 0 && in.worker.LastKnownVRAM < VRAMSoftLimitMB {
		vramPenalty = (1.0 - (float64(in.worker.LastKnownVRAM) / float64(in.totalVRAM))) * 1000.0
	}

	cacheBonus := 0.0
	ph := prefixHash(in.req.Data)
	if cachedID, ok := in.prefixCache[ph]; ok && cachedID == in.worker.ID {
		cacheBonus = CacheBonusMs
	}

	score = waitTime + execTime + tierPenalty + vramPenalty - cacheBonus
	breakdown = RoutingDecision{
		WorkerID:     in.worker.ID,
		ExecMs:       execTime,
		WaitMs:       waitTime,
		TierCostMs:   tierPenalty,
		VRAMCostMs:   vramPenalty,
		CacheBonusMs: cacheBonus,
		TotalMs:      score,
		Tokens:       tokens,
	}
	return score, breakdown, false
}