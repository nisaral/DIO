package scheduler

import (
	"sync"
	pb "github.com/nisaral/dio/api/proto"
)

type Scheduler struct {
	mu      sync.Mutex
	workers []*pb.RegisterRequest
	current int
}

func NewScheduler() *Scheduler {
	return &Scheduler{
		workers: []*pb.RegisterRequest{},
		current: 0,
	}
}

// UpdateWorkers refreshes the list of available workers from the registry
func (s *Scheduler) UpdateWorkers(workers []*pb.RegisterRequest) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.workers = workers
}

// PickWorker selects the next worker in line (Round-Robin)
func (s *Scheduler) PickWorker() *pb.RegisterRequest {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.workers) == 0 {
		return nil
	}

	worker := s.workers[s.current]
	// Increment and wrap around using modulo
	s.current = (s.current + 1) % len(s.workers)
	
	return worker
}