package scheduler

import (
	"os"
	"strconv"
	"strings"
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

// AdmissionDisabled is true when DIO_ADMISSION_OFF=1 (Locust scheduler comparisons).
func AdmissionDisabled() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("DIO_ADMISSION_OFF")))
	return v == "1" || v == "true" || v == "yes"
}