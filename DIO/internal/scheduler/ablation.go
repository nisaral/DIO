package scheduler

import (
	"os"
	"strings"
)

// AblationFlags zero out cost terms for paper ablations without recompiling.
// Set DIO_ABLATION to one of: full, no_queue, no_vram, no_vram_hard, no_tier, no_cache, no_dual
// Or compose with env: DIO_DISABLE_QUEUE=1, DIO_DISABLE_VRAM=1, DIO_DISABLE_VRAM_HARD=1,
// DIO_DISABLE_TIER=1, DIO_DISABLE_CACHE=1, NLMS_MODE=SINGLE.
type AblationFlags struct {
	DisableQueue    bool
	DisableVRAMSoft bool
	DisableVRAMHard bool
	DisableTier     bool
	DisableCache    bool
	SingleTimescale bool // alias NLMS_MODE=SINGLE
	Name            string
}

func LoadAblationFlags() AblationFlags {
	f := AblationFlags{Name: "full"}
	abl := strings.ToLower(strings.TrimSpace(os.Getenv("DIO_ABLATION")))
	switch abl {
	case "no_queue", "-queue", "minus_queue":
		f.DisableQueue = true
		f.Name = "no_queue"
	case "no_vram", "-vram", "minus_vram":
		f.DisableVRAMSoft = true
		f.DisableVRAMHard = true
		f.Name = "no_vram"
	case "no_vram_hard":
		f.DisableVRAMHard = true
		f.Name = "no_vram_hard"
	case "no_tier", "-tiers", "minus_tier":
		f.DisableTier = true
		f.Name = "no_tier"
	case "no_cache", "-cache":
		f.DisableCache = true
		f.Name = "no_cache"
	case "no_dual", "single", "single_mu":
		f.SingleTimescale = true
		f.Name = "no_dual"
	case "", "full":
		f.Name = "full"
	default:
		f.Name = abl
	}

	if envTruthy("DIO_DISABLE_QUEUE") {
		f.DisableQueue = true
	}
	if envTruthy("DIO_DISABLE_VRAM") {
		f.DisableVRAMSoft = true
		f.DisableVRAMHard = true
	}
	if envTruthy("DIO_DISABLE_VRAM_HARD") {
		f.DisableVRAMHard = true
	}
	if envTruthy("DIO_DISABLE_TIER") {
		f.DisableTier = true
	}
	if envTruthy("DIO_DISABLE_CACHE") {
		f.DisableCache = true
	}
	mode := strings.ToUpper(strings.TrimSpace(os.Getenv("NLMS_MODE")))
	if mode == "SINGLE" || mode == "SINGLE_MU" || mode == "1" {
		f.SingleTimescale = true
	}
	return f
}

func envTruthy(k string) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(k)))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}
