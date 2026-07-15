package scheduler

// NLMS hyperparameters (single source of truth — matches paper).
const (
	MuFast           = 0.1
	MuSlow           = 0.01
	MuBias           = 0.005
	FastSlowBlend    = 0.8 // weight on fast slope in effective slope (dual mode)
	TierMismatchMs   = 500.0
	CacheBonusMs     = 200.0
	VRAMSoftLimitMB  = 4096
	VRAMHardLimitMB  = 2400
	BatchSize        = 8.0
	SLOTTFTMs        = 2000.0
	// Mildly higher cold-start priors for real engines (online NLMS still adapts).
	// Absolute MAPE can remain large; routing depends on relative worker costs.
	InitialSlope     = 2.0
	InitialIntercept = 150.0
	// StaticProfile defaults used when SCHEDULER_STRATEGY=STATIC (offline calib baseline).
	StaticDefaultSlope     = 1.0
	StaticDefaultIntercept = 50.0
	// PredHistoryCap is max online prediction samples retained for MAPE export.
	PredHistoryCap = 5000
)

// RLS forgetting factor (lambda close to 1 for slow drift tracking).
const RLSLambda = 0.99