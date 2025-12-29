package worker_mgmt

import (
	"context"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
)

type DockerManager struct {
	cli *client.Client
}

func NewDockerManager() (*DockerManager, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv)
	if err != nil {
		return nil, err
	}
	return &DockerManager{cli: cli}, nil
}

// SpawnWorker spins up a new Python container from your worker image
func (dm *DockerManager) SpawnWorker(ctx context.Context, imageName string) error {
	resp, err := dm.cli.ContainerCreate(ctx, &container.Config{
		Image: imageName,
	}, nil, nil, nil, "")
	if err != nil {
		return err
	}

	return dm.cli.ContainerStart(ctx, resp.ID, container.StartOptions{})
}