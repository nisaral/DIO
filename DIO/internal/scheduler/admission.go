package scheduler

import "fmt"

// AdmissionError is returned when all workers are overloaded or memory-blocked.
type AdmissionError struct {
	RetryAfterSec int
	Message       string
}

func (e *AdmissionError) Error() string {
	return e.Message
}

func newAdmissionError(retryAfterMs float64, reason string) *AdmissionError {
	sec := int(retryAfterMs / 1000.0)
	if sec < 1 {
		sec = 1
	}
	return &AdmissionError{
		RetryAfterSec: sec,
		Message:       fmt.Sprintf("admission rejected: %s", reason),
	}
}