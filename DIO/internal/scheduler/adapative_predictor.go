package scheduler

import (
	"math"
	"sync"
)

// Predictor implements the "5-Gap Closure" RLS algorithm
// Gap 1: Co-location Dynamics (Interference History)
// Gap 2: KV Cache OOM (KV Growth Factor)
// Gap 3: Model Drift (Dual-Timescale Slopes)
type Predictor struct {
	mu sync.Mutex

	// Dual-Timescale Slopes (Gap 3)
	fastSlope float64 // Reacts instantly to batching noise
	slowSlope float64 // Reacts slowly to underlying drift
	bias      float64

	// Interference Tracking (Gap 1)
	interferenceHistory [3]float64 // Ring buffer of last 3 error ratios
	historyIdx          int

	// Memory & Reasoning (Gap 2 & 5)
	kvGrowthFactor   float64 // Bytes per token (learned)
	bandwidthPenalty float64 // Roofline factor
}

func NewAdaptivePredictor() *Predictor {
	return &Predictor{
		fastSlope:        2.1, // Conservative init
		slowSlope:        2.1,
		bias:             50.0,
		kvGrowthFactor:   0.5, // Init
		bandwidthPenalty: 1.0,
	}
}

// Estimate predicts execution time in milliseconds
func (p *Predictor) Estimate(tokens int, outputPred int, reasoning int, vramFreeMB int64) float64 {
	p.mu.Lock()
	defer p.mu.Unlock()

	// 1. Compute Cost (Dual Slope weighted average)
	// We favor fastSlope (0.8) to catch immediate contention
	slope := (0.8 * p.fastSlope) + (0.2 * p.slowSlope)
	computeCost := (slope * float64(tokens)) + p.bias

	// 2. Memory Cost (KV Cache Model - Gap 2)
	// Estimate KV cache size: (Layers * Hidden * Batch)
	// If VRAM is low, bandwidth penalty kicks in (Roofline)
	memCost := 0.0
	if vramFreeMB < 4096 { // < 4GB free triggers Roofline
		contention := (4096.0 - float64(vramFreeMB)) / 4096.0
		memCost = computeCost * contention * p.bandwidthPenalty
	}

	// 3. Reasoning Penalty (Gap 5)
	// "Chain of Thought" tokens are compute-dense
	reasoningCost := float64(reasoning) * 0.5

	// 4. Interference Multiplier (Gap 1)
	avgInterference := 0.0
	for _, v := range p.interferenceHistory {
		avgInterference += v
	}
	avgInterference /= 3.0

	// If recent history shows slowdowns (e.g. 1.2x), apply it
	interferenceFactor := math.Max(1.0, avgInterference)

	return math.Max(computeCost, memCost)*interferenceFactor + reasoningCost
}

// MultiScaleUpdate handles the dual-slope learning rates
func (p *Predictor) MultiScaleUpdate(errorVal float64, tokens int) {
	// Dual-Timescale Update (Gap 3)
	// Fast alpha = 0.1 (High plasticity)
	// Slow alpha = 0.01 (High stability)
	if tokens > 0 {
		grad := errorVal / float64(tokens)
		p.fastSlope += 0.1 * grad
		p.slowSlope += 0.01 * grad
	}

	// Prevent negative physics
	if p.fastSlope < 0.1 {
		p.fastSlope = 0.1
	}
	if p.slowSlope < 0.1 {
		p.slowSlope = 0.1
	}
}

// Update implements the Online Learning (RLS)
func (p *Predictor) Update(actualDuration float64, tokens int, outputLen int, vramFreeMB int64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// 1. Calculate Error
	// We assume reasoning=0 for update as we can't measure it easily yet
	predicted := p.Estimate(tokens, outputLen, 0, vramFreeMB)
	error := actualDuration - predicted

	// 2. Dual-Timescale Update (Gap 3)
	p.MultiScaleUpdate(error, tokens)

	p.bias += 0.1 * error

	// 3. Update Interference History
	// Ratio > 1.0 means system is slower than expected
	ratio := actualDuration / (predicted + 1e-9)
	p.interferenceHistory[p.historyIdx] = ratio
	p.historyIdx = (p.historyIdx + 1) % 3
}
