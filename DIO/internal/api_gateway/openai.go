package api_gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	pb "github.com/nisaral/dio/api/proto"
	"github.com/nisaral/dio/internal/scheduler"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream"`
}

type chatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []chatCompletionChoice `json:"choices"`
	Usage   chatUsage              `json:"usage"`
}

type chatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func writeSchedulerError(w http.ResponseWriter, err error) {
	if adm, ok := err.(*scheduler.AdmissionError); ok {
		w.Header().Set("Retry-After", strconv.Itoa(adm.RetryAfterSec))
		http.Error(w, adm.Error(), http.StatusServiceUnavailable)
		return
	}
	http.Error(w, err.Error(), http.StatusInternalServerError)
}

func extractPrompt(messages []chatMessage) string {
	if len(messages) == 0 {
		return ""
	}
	var parts []string
	for _, m := range messages {
		parts = append(parts, m.Role+": "+m.Content)
	}
	return strings.Join(parts, "\n")
}

// HandleChatCompletions implements POST /v1/chat/completions (non-streaming).
func HandleChatCompletions(managerAddr string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-DIO-Tier, X-DIO-Session-Id")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "invalid body", http.StatusBadRequest)
			return
		}

		var req chatCompletionRequest
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if req.Stream {
			http.Error(w, "streaming not yet supported; omit stream or set false", http.StatusNotImplemented)
			return
		}

		tier := r.Header.Get("X-DIO-Tier")
		if tier == "" {
			tier = "small"
		}

		prompt := extractPrompt(req.Messages)
		if prompt == "" {
			http.Error(w, "messages required", http.StatusBadRequest)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
		defer cancel()

		conn, err := grpc.NewClient(managerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			http.Error(w, "manager unavailable", http.StatusServiceUnavailable)
			return
		}
		defer conn.Close()

		client := pb.NewOrchestratorClient(conn)
		resp, err := client.ExecuteInference(ctx, &pb.InferenceRequest{
			ModelId: req.Model,
			Data:    []byte(prompt),
			Tier:    tier,
		})
		if err != nil {
			writeSchedulerError(w, err)
			return
		}

		content := string(resp.Output)
		if content == "" {
			content = "(empty response)"
		}

		promptTok := int(resp.PromptTokens)
		if promptTok == 0 {
			promptTok = len(prompt) / 4
		}
		compTok := int(resp.CompletionTokens)
		if compTok == 0 && resp.TokensUsed > 0 {
			compTok = int(resp.TokensUsed) - promptTok
			if compTok < 0 {
				compTok = int(resp.TokensUsed)
			}
		}

		out := chatCompletionResponse{
			ID:      "dio-chatcmpl",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   req.Model,
			Choices: []chatCompletionChoice{{
				Index: 0,
				Message: chatMessage{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: "stop",
			}},
			Usage: chatUsage{
				PromptTokens:     promptTok,
				CompletionTokens: compTok,
				TotalTokens:      promptTok + compTok,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(out)
	}
}