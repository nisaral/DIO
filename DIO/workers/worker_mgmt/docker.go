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

// SpawnWorker spins up a new Python container from your worker image
func (dm *DockerManager) SpawnWorker(ctx context.Context, imageName string) error {
	// Force the correct image name defined in docker-compose
	// The autoscaler might be passing "dio-python-worker" but we built it as "dio-worker"
	actualImage := "dio-worker"

	config := &container.Config{
		Image: actualImage,
		Env: []string{
			"MANAGER_ADDRESS=dio-manager:50052", // Tell worker where to find Manager
		},
		Labels: map[string]string{"type": "dio-worker"},
	}

	hostConfig := &container.HostConfig{
		NetworkMode: "dio_default", // Must match the docker-compose network name
		AutoRemove:  true,          // Clean up container when it exits
	}

	resp, err := dm.cli.ContainerCreate(ctx, config, hostConfig, nil, nil, "")
	if err != nil {
		return fmt.Errorf("failed to create worker container: %w", err)
	}

	log.Printf("Autoscaler: Spawning worker %s using image %s", resp.ID[:12], actualImage)
	return dm.cli.ContainerStart(ctx, resp.ID, container.StartOptions{})
}
