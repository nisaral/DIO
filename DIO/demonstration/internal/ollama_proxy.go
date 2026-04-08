package internal

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OllamaProxy handles chat requests via local Ollama instance
type OllamaProxy struct {
	baseURL      string // e.g., http://localhost:11434
	model        string // e.g., phi3:mini
	hub          *WebSocketHub
	systemPrompt string
}

func NewOllamaProxy(hub *WebSocketHub) *OllamaProxy {
	return &OllamaProxy{
		baseURL: "http://localhost:11434",
		model:   "mistral", // Changed from phi3:mini
		hub:     hub,
		systemPrompt: `You are a technical assistant for the DIO (Distributed Inference Orchestrator) system.
DIO is a production-grade LLM serving orchestrator that uses an adaptive NLMS (Normalized Least Mean Squares) scheduler.

Key facts about DIO:
- Written in Go (manager) + Python (workers) + gRPC (communication)
- NLMS predictor learns per-worker latency in O(1) time per update
- Dual-timescale filter: fast slope (burst adaptation) + slow slope (steady-state)
- Scoring function: predicted_latency + queue_penalty + vram_penalty
- Supports heterogeneous GPU clusters, auto-scaling via Docker
- Prevents Head-of-Line blocking via Shortest Job First (SJF) routing
- Claims: 2-3x better P99 latency vs Round Robin, O(1) scheduling overhead

Answer questions about DIO, NLMS, LLM serving, distributed systems, or ML infrastructure.
Keep answers concise and technical.`,
	}
}

type ChatRequest struct {
	Message string `json:"message"`
	Model   string `json:"model,omitempty"`
}

type ChatResponse struct {
	Response string `json:"response"`
	Model    string `json:"model"`
	Duration string `json:"duration"`
}

// Chat sends a message to Ollama and returns the response
func (op *OllamaProxy) Chat(message string) (*ChatResponse, error) {
	model := op.model

	op.hub.Broadcast(WSMessage{
		Type: "log",
		Data: map[string]string{
			"level":     "info",
			"message":   fmt.Sprintf("💬 Chat → Ollama (%s): \"%s\"", model, truncate(message, 60)),
			"timestamp": time.Now().Format("15:04:05"),
		},
	})

	payload := map[string]interface{}{
		"model":  model,
		"prompt": op.systemPrompt + "\n\nUser: " + message + "\n\nAssistant:",
		"stream": false,
		"options": map[string]interface{}{
			"num_predict": 300,
			"temperature": 0.7,
		},
	}

	body, _ := json.Marshal(payload)
	start := time.Now()
	resp, err := http.Post(op.baseURL+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama connection failed: %v", err)
	}
	defer resp.Body.Close()
	duration := time.Since(start)

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("ollama error (HTTP %d): %s", resp.StatusCode, string(respBody))
	}

	var ollamaResp struct {
		Response string `json:"response"`
		Model    string `json:"model"`
	}
	if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to parse ollama response: %v", err)
	}

	// Clean response
	response := strings.TrimSpace(ollamaResp.Response)

	result := &ChatResponse{
		Response: response,
		Model:    ollamaResp.Model,
		Duration: duration.Round(time.Millisecond).String(),
	}

	op.hub.Broadcast(WSMessage{
		Type: "log",
		Data: map[string]string{
			"level":     "success",
			"message":   fmt.Sprintf("💬 Ollama responded in %s (%d chars)", result.Duration, len(response)),
			"timestamp": time.Now().Format("15:04:05"),
		},
	})

	return result, nil
}

// ListModels returns available Ollama models
func (op *OllamaProxy) ListModels() ([]string, error) {
	resp, err := http.Get(op.baseURL + "/api/tags")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var data struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}

	names := make([]string, len(data.Models))
	for i, m := range data.Models {
		names[i] = m.Name
	}
	return names, nil
}

// SetModel changes the active model
func (op *OllamaProxy) SetModel(model string) {
	op.model = model
}
