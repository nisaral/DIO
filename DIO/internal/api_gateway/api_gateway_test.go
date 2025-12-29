package apigateway_test

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAPIGateway(t *testing.T) {
	req, err := http.NewRequest("GET", "/api/test", nil)
	assert.NoError(t, err)

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(APIGatewayHandler) // Replace with your actual handler function

	handler.ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code)
	assert.JSONEq(t, `{"message": "success"}`, rr.Body.String()) // Replace with expected response
}