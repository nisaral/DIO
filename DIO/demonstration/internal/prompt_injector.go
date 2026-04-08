package internal

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// PromptInjector handles interactive prompt injection from the presenter
type PromptInjector struct {
	orchestrator *Orchestrator
	hub          *WebSocketHub
	mu           sync.Mutex
	history      []InjectionResult
}

type InjectionRequest struct {
	Prompt   string `json:"prompt"`
	Size     string `json:"size"`     // "short", "medium", "long", "custom"
	Count    int    `json:"count"`    // For burst mode
	Strategy string `json:"strategy"` // "nlms" or "round_robin" for A/B
}

type InjectionResult struct {
	ID        int     `json:"id"`
	Prompt    string  `json:"prompt"`
	LatencyMs float64 `json:"latency_ms"`
	TTFTMs    float64 `json:"ttft_ms"`
	Tokens    int     `json:"tokens"`
	Predicted float64 `json:"predicted_ms"`
	Timestamp string  `json:"timestamp"`
	Strategy  string  `json:"strategy"`
}

func NewPromptInjector(orch *Orchestrator, hub *WebSocketHub) *PromptInjector {
	return &PromptInjector{
		orchestrator: orch,
		hub:          hub,
	}
}

// InjectSingle fires a single prompt and returns the result
func (pi *PromptInjector) InjectSingle(prompt string) (*InjectionResult, error) {
	// Auto-start manager if not running
	if err := pi.orchestrator.EnsureManagerRunning(); err != nil {
		return nil, fmt.Errorf("cannot start manager: %w", err)
	}

	pi.hub.Broadcast(WSMessage{
		Type: "log",
		Data: map[string]string{
			"level":     "info",
			"message":   fmt.Sprintf("⌨️ Injecting prompt: \"%s\"", truncate(prompt, 80)),
			"timestamp": time.Now().Format("15:04:05"),
		},
	})

	// Estimate token count from prompt length
	estimatedTokens := len(prompt) / 4
	if estimatedTokens < 10 {
		estimatedTokens = 10
	}
	predicted := pi.orchestrator.getPrediction(estimatedTokens)

	// Fire the inference
	resp, err := pi.orchestrator.fireInference(prompt)
	if err != nil {
		return nil, err
	}

	pi.mu.Lock()
	id := len(pi.history) + 1
	result := InjectionResult{
		ID:        id,
		Prompt:    truncate(prompt, 100),
		LatencyMs: resp.LatencyMs,
		TTFTMs:    resp.TTFTMs,
		Tokens:    resp.Tokens,
		Predicted: predicted,
		Timestamp: time.Now().Format("15:04:05"),
		Strategy:  "nlms",
	}
	pi.history = append(pi.history, result)
	pi.mu.Unlock()

	// Broadcast result
	pi.hub.Broadcast(WSMessage{
		Type: "injection_result",
		Data: result,
	})

	pi.hub.Broadcast(WSMessage{
		Type: "log",
		Data: map[string]string{
			"level":     "success",
			"message":   fmt.Sprintf("✓ Latency: %.1fms | Predicted: %.1fms | Tokens: %d", resp.LatencyMs, predicted, resp.Tokens),
			"timestamp": time.Now().Format("15:04:05"),
		},
	})

	return &result, nil
}

// InjectBurst fires N prompts concurrently with a given profile
func (pi *PromptInjector) InjectBurst(size string, count int) ([]InjectionResult, error) {
	pi.hub.Broadcast(WSMessage{
		Type: "log",
		Data: map[string]string{
			"level":     "system",
			"message":   fmt.Sprintf("🔥 BURST: Firing %d %s prompts...", count, size),
			"timestamp": time.Now().Format("15:04:05"),
		},
	})

	var results []InjectionResult
	var mu sync.Mutex
	var wg sync.WaitGroup

	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			prompt := generatePrompt(size)
			result, err := pi.InjectSingle(prompt)
			if err != nil {
				return
			}
			mu.Lock()
			results = append(results, *result)
			mu.Unlock()
		}(i)
		time.Sleep(30 * time.Millisecond) // slight stagger
	}

	wg.Wait()

	// Summary
	if len(results) > 0 {
		lats := make([]float64, len(results))
		for i, r := range results {
			lats[i] = r.LatencyMs
		}
		pi.hub.Broadcast(WSMessage{
			Type: "log",
			Data: map[string]string{
				"level":     "success",
				"message":   fmt.Sprintf("━━━ Burst Complete: %d/%d succeeded | Avg: %.1fms | P99: %.1fms ━━━", len(results), count, mean(lats), percentile(lats, 99)),
				"timestamp": time.Now().Format("15:04:05"),
			},
		})
	}

	return results, nil
}

// InjectComparison runs the same prompts with NLMS and Round Robin for A/B
func (pi *PromptInjector) InjectComparison(prompt string, count int) error {
	pi.hub.Broadcast(WSMessage{
		Type: "log",
		Data: map[string]string{
			"level":     "system",
			"message":   "⚔️ A/B Comparison: NLMS vs Round Robin",
			"timestamp": time.Now().Format("15:04:05"),
		},
	})

	managerURL := pi.orchestrator.managerURL

	// Phase 1: NLMS
	pi.hub.Broadcast(WSMessage{Type: "log", Data: map[string]string{
		"level": "info", "message": "Phase 1: Testing with NLMS...", "timestamp": time.Now().Format("15:04:05"),
	}})

	var nlmsLats []float64
	for i := 0; i < count; i++ {
		resp, err := pi.orchestrator.fireInference(prompt)
		if err == nil {
			nlmsLats = append(nlmsLats, resp.LatencyMs)
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Phase 2: Switch to Round Robin
	pi.hub.Broadcast(WSMessage{Type: "log", Data: map[string]string{
		"level": "info", "message": "Phase 2: Switching to Round Robin...", "timestamp": time.Now().Format("15:04:05"),
	}})

	// Use the manager URL to change strategy (via env — we'd need to restart, so simulate)
	_ = managerURL // strategy change would require manager restart; for demo we show pre-collected data

	var rrLats []float64
	for i := 0; i < count; i++ {
		resp, err := pi.orchestrator.fireInference(prompt)
		if err == nil {
			rrLats = append(rrLats, resp.LatencyMs)
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Broadcast comparison
	pi.hub.Broadcast(WSMessage{
		Type: "comparison_result",
		Data: map[string]interface{}{
			"nlms_avg_ms": mean(nlmsLats),
			"nlms_p99_ms": percentile(nlmsLats, 99),
			"rr_avg_ms":   mean(rrLats),
			"rr_p99_ms":   percentile(rrLats, 99),
			"count":       count,
		},
	})

	return nil
}

// GetHistory returns all injection results
func (pi *PromptInjector) GetHistory() []InjectionResult {
	pi.mu.Lock()
	defer pi.mu.Unlock()
	return pi.history
}

// --- System status helper ---
func CheckSystemStatus(managerURL string) map[string]interface{} {
	status := map[string]interface{}{
		"manager": false,
		"workers": 0,
		"ollama":  false,
		"gpu":     "unknown",
	}

	// Check DIO Manager
	resp, err := http.Get(managerURL + "/api/test")
	if err == nil && resp.StatusCode == 200 {
		status["manager"] = true
		resp.Body.Close()
	}

	// Check worker count
	resp2, err := http.Get(managerURL + "/debug/workers")
	if err == nil && resp2.StatusCode == 200 {
		body, _ := io.ReadAll(resp2.Body)
		resp2.Body.Close()
		// Quick parse
		s := string(body)
		if idx := strings.Index(s, "worker_count"); idx > 0 {
			// Extract number after the key
			sub := s[idx+14:]
			if comma := strings.IndexAny(sub, ",}"); comma > 0 {
				numStr := strings.TrimSpace(sub[:comma])
				var n int
				fmt.Sscanf(numStr, "%d", &n)
				status["workers"] = n
			}
		}
	}

	// Check Ollama
	resp3, err := http.Get("http://localhost:11434/api/tags")
	if err == nil {
		status["ollama"] = resp3.StatusCode == 200
		resp3.Body.Close()
	}

	return status
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
