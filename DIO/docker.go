package worker_mgmt

import (
	"context"
	"fmt"
	"log"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
)

type DockerManager struct {
	cli *client.Client
}

func NewDockerManager() (*DockerManager, error) {
	// Pin Docker API version to 1.47 to avoid "client version is too new" errors
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithVersion("1.47"))
	if err != nil {
		return nil, err
	}
	return &DockerManager{cli: cli}, nil
}

// LaunchWorker starts a new worker container
func (dm *DockerManager) LaunchWorker(ctx context.Context) error {
	// Define container configuration
	config := &container.Config{
		Image: "dio-worker", // Assumes the worker image is named 'dio-worker'
		Env: []string{
			"MANAGER_ADDRESS=dio-manager:50052", // Use internal gRPC port
		},
		Labels: map[string]string{"type": "dio-worker"},
	}

	hostConfig := &container.HostConfig{
		NetworkMode: "dio_default", // Assumes the docker-compose network name
		AutoRemove:  true,
	}

	resp, err := dm.cli.ContainerCreate(ctx, config, hostConfig, nil, nil, "")
	if err != nil {
		return fmt.Errorf("failed to create worker container: %w", err)
	}

	if err := dm.cli.ContainerStart(ctx, resp.ID, container.StartOptions{}); err != nil {
		return fmt.Errorf("failed to start worker container: %w", err)
	}

	log.Printf("Autoscaler: Started new worker %s", resp.ID[:12])
	return nil
}

// CountWorkers returns the number of active worker containers
func (dm *DockerManager) CountWorkers(ctx context.Context) (int, error) {
	containers, err := dm.cli.ContainerList(ctx, container.ListOptions{})
	if err != nil {
		return 0, err
	}

	count := 0
	for _, c := range containers {
		// Check labels or image name
		if c.Labels["type"] == "dio-worker" {
			count++
		}
	}
	return count, nil
}
