package internal

import (
	"log"
	"sync"

	"github.com/gorilla/websocket"
)

// WSMessage is the standard message format sent over WebSocket
type WSMessage struct {
	Type string      `json:"type"` // "log", "metrics_update", "test_complete", "chat_response", "system_status"
	Data interface{} `json:"data"`
}

// WebSocketHub manages connected WebSocket clients
type WebSocketHub struct {
	mu      sync.Mutex
	clients map[*websocket.Conn]bool
}

func NewWebSocketHub() *WebSocketHub {
	return &WebSocketHub{
		clients: make(map[*websocket.Conn]bool),
	}
}

func (h *WebSocketHub) Register(conn *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.clients[conn] = true
	log.Printf("[WS] Client connected (%d total)", len(h.clients))
}

func (h *WebSocketHub) Unregister(conn *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.clients, conn)
	conn.Close()
	log.Printf("[WS] Client disconnected (%d remaining)", len(h.clients))
}

func (h *WebSocketHub) Broadcast(msg WSMessage) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for conn := range h.clients {
		if err := conn.WriteJSON(msg); err != nil {
			log.Printf("[WS] Write error: %v", err)
			delete(h.clients, conn)
			conn.Close()
		}
	}
}

func (h *WebSocketHub) ClientCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.clients)
}
