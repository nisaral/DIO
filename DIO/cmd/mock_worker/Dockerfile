FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /mock-worker ./cmd/mock_worker/main.go

FROM alpine:3.19
WORKDIR /app
COPY --from=build /mock-worker .
EXPOSE 50060 50061
ENTRYPOINT ["/app/mock-worker"]