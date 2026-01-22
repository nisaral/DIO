package registry

import (
	pb "github.com/nisaral/dio/api/proto"
)

type Worker struct {
	ID            string
	Address       string
	IsHealthy     bool
	LastKnownVRAM int64
	// TaskChannel is the "Queue" for this specific worker
	// We use its length to calculate Head-of-Line wait time
	TaskChannel chan *pb.InferenceRequest
}
