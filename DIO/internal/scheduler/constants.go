package scheduler

// NLMS hyperparameters (single source of truth — matches paper).
const (
	MuFast          = 0.1
	MuSlow          = 0.01
	MuBias          = 0.005
	FastSlowBlend   = 0.8 // weight on fast slope in effective slope
	TierMismatchMs  = 500.0
	CacheBonusMs    = 200.0
	VRAMSoftLimitMB = 4096
	VRAMHardLimitMB = 2400
	BatchSize       = 8.0
	SLOTTFTMs       = 2000.0
	InitialSlope    = 0.1
	InitialIntercept = 50.0
)

// RLS forgetting factor (lambda close to 1 for slow drift tracking).
const RLSLambda = 0.99