#!/bin/bash
# Helper script to run MedImages.jl Docker container
# Usage: ./scripts/docker-run.sh [command] [args...]
#
# Examples:
#   ./scripts/docker-run.sh                    # Start Julia REPL
#   ./scripts/docker-run.sh julia -e "..."     # Run Julia command
#   ./scripts/docker-run.sh bash               # Start bash shell

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="medimages:latest"

cd "$PROJECT_DIR"

# Check if image exists, build if not
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Docker image not found, building..."
    docker compose build medimages
fi

# Default to interactive shell if no command given
if [ $# -eq 0 ]; then
    echo "Starting interactive Julia REPL..."
    docker compose run --rm medimages
else
    docker compose run --rm medimages "$@"
fi
