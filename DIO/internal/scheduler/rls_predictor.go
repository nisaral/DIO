package scheduler

import (
	"math"
	"sync"
)

// PerWorkerRLSPredictor implements Recursive Least Squares for latency = slope*tokens + intercept.
// Uses a 2x2 covariance matrix P (O(4) per update; scales as O(d^2) with feature dimension d).
type PerWorkerRLSPredictor struct {
	mu             sync.Mutex
	Slope          float64
	Intercept      float64
	P              [2][2]float64 // covariance of [slope, intercept]
	AverageLatency float64
	UpdateCount    int
	Tier           string
	TotalVRAM      int64
	EngineType     string
}

func NewRLSPredictor(tier string, totalVRAM int64, engineType string) *PerWorkerRLSPredictor {
	return &PerWorkerRLSPredictor{
		Slope:     InitialSlope,
		Intercept: InitialIntercept,
		P: [2][2]float64{
			{1000.0, 0},
			{0, 1000.0},
		},
		Tier:       tier,
		TotalVRAM:  totalVRAM,
		EngineType: engineType,
	}
}

func (p *PerWorkerRLSPredictor) Update(actual float64, tokens int, _ int64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if tokens <= 0 {
		return
	}

	x0 := float64(tokens)
	x1 := 1.0
	predicted := p.Slope*x0 + p.Intercept
	err := actual - predicted

	// P*x
	px0 := p.P[0][0]*x0 + p.P[0][1]*x1
	px1 := p.P[1][0]*x0 + p.P[1][1]*x1

	denom := RLSLambda + x0*px0 + x1*px1
	if denom < 1e-9 {
		denom = 1e-9
	}
	k0 := px0 / denom
	k1 := px1 / denom

	p.Slope += k0 * err
	p.Intercept += k1 * err

	// P = (P - K*x^T*P) / lambda
	p00 := p.P[0][0]
	p01 := p.P[0][1]
	p10 := p.P[1][0]
	p11 := p.P[1][1]

	p.P[0][0] = (p00 - k0*(x0*p00+x1*p10)) / RLSLambda
	p.P[0][1] = (p01 - k0*(x0*p01+x1*p11)) / RLSLambda
	p.P[1][0] = (p10 - k1*(x0*p00+x1*p10)) / RLSLambda
	p.P[1][1] = (p11 - k1*(x0*p01+x1*p11)) / RLSLambda

	if p.Slope < InitialSlope {
		p.Slope = InitialSlope
	}
	if p.Intercept < 0.1 {
		p.Intercept = 0.1
	}

	p.UpdateCount++
	if p.AverageLatency == 0 {
		p.AverageLatency = actual
	} else {
		p.AverageLatency = 0.9*p.AverageLatency + 0.1*actual
	}
}

func (p *PerWorkerRLSPredictor) Estimate(tokens int, vramMB int64) (float64, float64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if vramMB < VRAMHardLimitMB && tokens > 1000 {
		return math.MaxFloat64, p.AverageLatency
	}

	base := p.Slope*float64(tokens) + p.Intercept
	return base, p.AverageLatency
}

func (p *PerWorkerRLSPredictor) Snapshot() PredictorSnapshot {
	p.mu.Lock()
	defer p.mu.Unlock()
	return PredictorSnapshot{
		FastSlope: p.Slope,
		SlowSlope: p.Slope,
		Intercept: p.Intercept,
		AvgLatency: p.AverageLatency,
		Updates:   p.UpdateCount,
		Algorithm: "RLS",
	}
}