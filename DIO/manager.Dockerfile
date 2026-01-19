FROM golang:1.24-alpine AS builder

WORKDIR /app

# Copy go.mod and download dependencies
COPY go.mod ./
# If you have a go.sum, uncomment the next line
# COPY go.sum ./
RUN go mod download

COPY . .

RUN go build -o manager cmd/manager/main.go

FROM alpine:latest

WORKDIR /root/
COPY --from=builder /app/manager .

EXPOSE 50051

CMD ["./manager"]