package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"

	"github.com/gorilla/websocket"
	"github.com/nisaral/dio/demonstration/internal"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func main() {
	log.SetFlags(log.Ltime | log.Lshortfile)
	log.Println("╔═══════════════════════════════════════════╗")
	log.Println("║   DIO Demonstration Server v1.0           ║")
	log.Println("║   http://localhost:9090                    ║")
	log.Println("╚═══════════════════════════════════════════╝")

	// Find project root (the DIO directory)
	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)
	// If running from demonstration/cmd/demo, project root is 3 levels up
	dioRoot := filepath.Join(exeDir, "..", "..", "..")

	// Override with env if set
	if env := os.Getenv("DIO_ROOT"); env != "" {
		dioRoot = env
	}

	// Resolve absolute path
	dioRoot, _ = filepath.Abs(dioRoot)

	managerBin := filepath.Join(dioRoot, "dio-manager")
	workerBin := filepath.Join(dioRoot, "mock-worker")
	if runtime.GOOS == "windows" {
		managerBin += ".exe"
		workerBin += ".exe"
	}
	frontendDir := filepath.Join(dioRoot, "demonstration", "frontend")

	log.Printf("DIO Root:     %s", dioRoot)
	log.Printf("Manager Bin:  %s", managerBin)
	log.Printf("Worker Bin:   %s", workerBin)
	log.Printf("Frontend Dir: %s", frontendDir)

	// Initialize components
	hub := internal.NewWebSocketHub()
	orch := internal.NewOrchestrator("http://localhost:8085", managerBin, workerBin, hub)
	injector := internal.NewPromptInjector(orch, hub)
	ollama := internal.NewOllamaProxy(hub)

	// ═══════════════════════════════════════════
	// HTTP Routes
	// ═══════════════════════════════════════════

	// Serve frontend static files
	fs := http.FileServer(http.Dir(frontendDir))
	http.Handle("/", fs)

	// WebSocket endpoint
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("WebSocket upgrade failed: %v", err)
			return
		}
		hub.Register(conn)
		defer hub.Unregister(conn)

		// Read loop (handles incoming messages from UI)
		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				break
			}
			var wsMsg struct {
				Type string          `json:"type"`
				Data json.RawMessage `json:"data"`
			}
			if err := json.Unmarshal(msg, &wsMsg); err != nil {
				continue
			}

			switch wsMsg.Type {
			case "chat":
				var chatReq internal.ChatRequest
				json.Unmarshal(wsMsg.Data, &chatReq)
				go func() {
					resp, err := ollama.Chat(chatReq.Message)
					if err != nil {
						hub.Broadcast(internal.WSMessage{
							Type: "chat_response",
							Data: map[string]string{"error": err.Error()},
						})
						return
					}
					hub.Broadcast(internal.WSMessage{Type: "chat_response", Data: resp})
				}()
			}
		}
	})

	// API: List tests
	http.HandleFunc("/api/tests", cors(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(internal.GetTestDefinitions())
	}))

	// API: Run single test
	http.HandleFunc("/api/tests/run", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			TestID string `json:"test_id"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		go func() {
			result, err := orch.RunTest(req.TestID)
			if err != nil {
				log.Printf("Test %s failed: %v", req.TestID, err)
			} else {
				log.Printf("Test %s completed: %s", req.TestID, result.Status)
			}
		}()
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{"status": "started", "test_id": req.TestID})
	}))

	// API: Run all tests
	http.HandleFunc("/api/tests/run-all", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		go orch.RunAllTests()
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{"status": "started"})
	}))

	// API: Inject single prompt
	http.HandleFunc("/api/inject", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req internal.InjectionRequest
		json.NewDecoder(r.Body).Decode(&req)

		prompt := req.Prompt
		if prompt == "" {
			prompt = internal.GeneratePromptExported(req.Size)
		}

		go func() {
			result, err := injector.InjectSingle(prompt)
			if err != nil {
				log.Printf("Injection failed: %v", err)
			} else {
				log.Printf("Injected: latency=%.1fms tokens=%d", result.LatencyMs, result.Tokens)
			}
		}()
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{"status": "injecting"})
	}))

	// API: Burst inject
	http.HandleFunc("/api/inject/burst", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req internal.InjectionRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Count == 0 {
			req.Count = 10
		}
		if req.Size == "" {
			req.Size = "short"
		}
		go injector.InjectBurst(req.Size, req.Count)
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{"status": "burst_started", "count": fmt.Sprintf("%d", req.Count)})
	}))

	// API: System status
	http.HandleFunc("/api/system/status", cors(func(w http.ResponseWriter, r *http.Request) {
		status := internal.CheckSystemStatus("http://localhost:8085")
		status["ws_clients"] = hub.ClientCount()
		json.NewEncoder(w).Encode(status)
	}))

	// API: Chat with Ollama
	http.HandleFunc("/api/chat", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req internal.ChatRequest
		json.NewDecoder(r.Body).Decode(&req)

		if req.Model != "" {
			ollama.SetModel(req.Model)
		}

		resp, err := ollama.Chat(req.Message)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(resp)
	}))

	// API: List Ollama models
	http.HandleFunc("/api/chat/models", cors(func(w http.ResponseWriter, r *http.Request) {
		models, err := ollama.ListModels()
		if err != nil {
			json.NewEncoder(w).Encode(map[string]interface{}{"models": []string{}, "error": err.Error()})
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"models": models})
	}))

	// API: Thermal Throttle (Chaos Engineering)
	http.HandleFunc("/api/worker/throttle", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			WorkerID   string  `json:"worker_id"`
			Multiplier float64 `json:"multiplier"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if req.WorkerID == "" {
			req.WorkerID = "RTX4050_Shard_2"
		}
		orch.SetThrottle(req.WorkerID, req.Multiplier)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":     "ok",
			"worker_id":  req.WorkerID,
			"multiplier": req.Multiplier,
		})
	}))

	// API: OOM Bomb (Chaos Engineering)
	http.HandleFunc("/api/chaos/oom-bomb", cors(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		go orch.FireOOMBomb()
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{"status": "fired"})
	}))

	// API: Injection history
	http.HandleFunc("/api/inject/history", cors(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(injector.GetHistory())
	}))

	// Start server
	port := os.Getenv("DEMO_PORT")
	if port == "" {
		port = "9090"
	}
	log.Printf("🚀 Demo server starting on :%s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

// cors wraps a handler with CORS headers
func cors(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}
