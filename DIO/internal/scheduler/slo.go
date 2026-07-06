package scheduler

import (
	"os"
	"strconv"
)

// EffectiveSLOMs returns the admission / routing SLO budget in milliseconds.
// Benchmarks set DIO_SLO_MS high (e.g. 90000) because minScore is end-to-end
// wait+exec for full generation, not TTFT-only.
func EffectiveSLOMs() float64 {
	if v := os.Getenv("DIO_SLO_MS"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			return f
		}
	}
	return SLOTTFTMs
}