#!/usr/bin/env bash
# Build and start ActiveCircuitDiscovery on DGX Spark
set -euo pipefail

IMAGE="activecircuitdiscovery:latest"
COMPOSE_FILE="docker-compose-dgx.yml"

echo "=== Building $IMAGE ==="
DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    --cache-from "$IMAGE" \
    -t "$IMAGE" \
    -f Dockerfile \
    .

echo "=== Starting JupyterLab service ==="
docker compose -f "$COMPOSE_FILE" up -d jupyterlab

echo ""
echo "=== Services ==="
echo "  JupyterLab:  http://localhost:8888"
echo "  Dashboard:   http://localhost:8050  (start with --profile dashboard)"
echo ""
echo "To run a full experiment:"
echo "  docker compose -f $COMPOSE_FILE --profile experiment up"
