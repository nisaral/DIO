package scheduler

import (
	"sync"
	"sync/atomic"
	"time"
)

// AdmissionStats tracks proactive admission as a goodput optimizer.
// Reject if min_w S_w > SLO (or hard VRAM block on all workers).
type AdmissionStats struct {
	Admitted            uint64 `json:"admitted"`
	RejectedSLO         uint64 `json:"rejected_slo"`
	RejectedVRAM        uint64 `json:"rejected_vram"`
	RejectedNoWorker    uint64 `json:"rejected_no_worker"`
	CompletedUnderSLO   uint64 `json:"completed_under_slo"`
	CompletedOverSLO    uint64 `json:"completed_over_slo"`
	CompletedTotal      uint64 `json:"completed_total"`
	SumE2EMs            uint64 `json:"sum_e2e_ms"` // integer ms for atomic add
	SumPredictedMs      uint64 `json:"sum_predicted_ms"`
}

// Snapshot returns a copy of counters.
func (a *AdmissionStats) Snapshot() map[string]interface{} {
	admitted := atomic.LoadUint64(&a.Admitted)
	rejSLO := atomic.LoadUint64(&a.RejectedSLO)
	rejVRAM := atomic.LoadUint64(&a.RejectedVRAM)
	rejNW := atomic.LoadUint64(&a.RejectedNoWorker)
	under := atomic.LoadUint64(&a.CompletedUnderSLO)
	over := atomic.LoadUint64(&a.CompletedOverSLO)
	total := atomic.LoadUint64(&a.CompletedTotal)
	sumE2E := atomic.LoadUint64(&a.SumE2EMs)
	sumPred := atomic.LoadUint64(&a.SumPredictedMs)

	goodputRate := 0.0
	if total > 0 {
		goodputRate = float64(under) / float64(total)
	}
	avgE2E := 0.0
	if total > 0 {
		avgE2E = float64(sumE2E) / float64(total)
	}
	// Goodput proxy: fraction of completions under SLO among attempts
	// (admitted + rejected). Venue metric: completed_under_slo / wall time
	// is computed by the suite; here we expose raw counters.
	return map[string]interface{}{
		"admitted":              admitted,
		"rejected_slo":          rejSLO,
		"rejected_vram":         rejVRAM,
		"rejected_no_worker":    rejNW,
		"completed_under_slo":   under,
		"completed_over_slo":    over,
		"completed_total":       total,
		"goodput_fraction":      goodputRate,
		"avg_e2e_ms":            avgE2E,
		"sum_predicted_ms":      sumPred,
		"slo_ms":                EffectiveSLOMs(),
		"admission_enabled":     !AdmissionDisabled(),
	}
}

func (a *AdmissionStats) RecordAdmit() {
	atomic.AddUint64(&a.Admitted, 1)
}

func (a *AdmissionStats) RecordRejectSLO() {
	atomic.AddUint64(&a.RejectedSLO, 1)
}

func (a *AdmissionStats) RecordRejectVRAM() {
	atomic.AddUint64(&a.RejectedVRAM, 1)
}

func (a *AdmissionStats) RecordRejectNoWorker() {
	atomic.AddUint64(&a.RejectedNoWorker, 1)
}

func (a *AdmissionStats) RecordCompletion(e2eMs, predictedMs float64) {
	atomic.AddUint64(&a.CompletedTotal, 1)
	if e2eMs > 0 {
		atomic.AddUint64(&a.SumE2EMs, uint64(e2eMs))
	}
	if predictedMs > 0 {
		atomic.AddUint64(&a.SumPredictedMs, uint64(predictedMs))
	}
	if e2eMs > 0 && e2eMs <= EffectiveSLOMs() {
		atomic.AddUint64(&a.CompletedUnderSLO, 1)
	} else if e2eMs > 0 {
		atomic.AddUint64(&a.CompletedOverSLO, 1)
	}
}

// PredSample is one online prediction vs actual (for dual-vs-single MAPE plots).
type PredSample struct {
	UnixMs    int64   `json:"unix_ms"`
	WorkerID  string  `json:"worker_id"`
	Tokens    int     `json:"tokens"`
	Predicted float64 `json:"predicted_ms"`
	Actual    float64 `json:"actual_ms"`
	AbsErr    float64 `json:"abs_err_ms"`
	RelErr    float64 `json:"rel_err"`
	FastSlope float64 `json:"fast_slope"`
	SlowSlope float64 `json:"slow_slope"`
	Mode      string  `json:"mode"` // DUAL | SINGLE
}

// PredHistory is a bounded ring buffer of prediction samples.
type PredHistory struct {
	mu      sync.Mutex
	samples []PredSample
	max     int
	sumAbs  float64
	sumRel  float64
	n       int
}

func NewPredHistory(max int) *PredHistory {
	if max <= 0 {
		max = 5000
	}
	return &PredHistory{samples: make([]PredSample, 0, max), max: max}
}

func (h *PredHistory) Add(s PredSample) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.samples) >= h.max {
		// drop oldest
		old := h.samples[0]
		h.sumAbs -= old.AbsErr
		h.sumRel -= old.RelErr
		h.samples = h.samples[1:]
		h.n--
	}
	h.samples = append(h.samples, s)
	h.sumAbs += s.AbsErr
	h.sumRel += s.RelErr
	h.n++
}

func (h *PredHistory) Snapshot(limit int) map[string]interface{} {
	h.mu.Lock()
	defer h.mu.Unlock()
	mape := 0.0
	mae := 0.0
	if h.n > 0 {
		mae = h.sumAbs / float64(h.n)
		mape = (h.sumRel / float64(h.n)) * 100.0
	}
	out := h.samples
	if limit > 0 && len(out) > limit {
		out = out[len(out)-limit:]
	}
	// copy
	cp := make([]PredSample, len(out))
	copy(cp, out)
	return map[string]interface{}{
		"count":   h.n,
		"mae_ms":  mae,
		"mape_pct": mape,
		"samples": cp,
	}
}

func nowUnixMs() int64 {
	return time.Now().UnixMilli()
}
