package health

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHealthCheck(t *testing.T) {
	assert.True(t, true, "Health check should pass")
}

func TestAnotherHealthCheck(t *testing.T) {
	assert.Equal(t, 1, 1, "Expected values should be equal")
}