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

// SpawnWorker spins up a new Python container from the dio-worker image.
func (dm *DockerManager) SpawnWorker(ctx context.Context, network, managerAddr string) error {
	actualImage := "dio-worker"
	if network == "" {
		network = "dio_default"
	}
	if managerAddr == "" {
		managerAddr = "dio-manager:50055"
	}

	config := &container.Config{
		Image: actualImage,
		Env: []string{
			"MANAGER_ADDRESS=" + managerAddr,
		},
		Labels: map[string]string{"type": "dio-worker"},
	}

	hostConfig := &container.HostConfig{
		NetworkMode: container.NetworkMode(network),
		AutoRemove:  true,
	}

	resp, err := dm.cli.ContainerCreate(ctx, config, hostConfig, nil, nil, "")
	if err != nil {
		return fmt.Errorf("failed to create worker container: %w", err)
	}

	log.Printf("Autoscaler: Spawning worker %s using image %s", resp.ID[:12], actualImage)
	return dm.cli.ContainerStart(ctx, resp.ID, container.StartOptions{})
}
