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

# Copy UI and dashboard assets
COPY ui ./ui
COPY dashboard ./dashboard

# Expose gRPC (50055) and HTTP (8085)
EXPOSE 50055 8085

CMD ["./manager"]