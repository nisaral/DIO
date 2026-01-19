# Build Stage
FROM golang:1.24-alpine AS builder

WORKDIR /app

# Copy module files first for caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the manager binary
RUN go build -o manager cmd/manager/main.go

# Runtime Stage
FROM alpine:latest
WORKDIR /root/

# Copy binary from builder
COPY --from=builder /app/manager .

# Copy UI assets (required by main.go)
COPY ui ./ui

# Expose gRPC (50052) and HTTP (8080)
EXPOSE 50052 8080

CMD ["./manager"]