#!/bin/bash
# Quick start script for MedImages.jl Docker environment
# Usage: ./scripts/quick-start.sh
#
# This script:
# 1. Builds the Docker image
# 2. Verifies CUDA and Python setup
# 3. Runs a quick test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "MedImages.jl Quick Start"
echo "========================"
echo ""

# Step 1: Build
echo "Step 1: Building Docker image..."
echo ""
docker compose build medimages
echo ""
echo "[OK] Docker image built successfully"
echo ""

# Step 2: Check CUDA
echo "Step 2: Checking CUDA availability..."
echo ""
docker compose run --rm medimages julia --project=. -e '
    using CUDA
    println("CUDA functional: ", CUDA.functional())
    if CUDA.functional()
        println("GPU: ", CUDA.name(CUDA.device()))
        println("Memory: ", round(CUDA.totalmem(CUDA.device()) / 1e9, digits=2), " GB")
    end
' || echo "[WARN] CUDA check failed - GPU may not be available"
echo ""

# Step 3: Check Python
echo "Step 3: Checking Python/SimpleITK..."
echo ""
docker compose run --rm medimages julia --project=. -e '
    using PyCall
    sitk = pyimport("SimpleITK")
    println("SimpleITK version: ", sitk.Version())
' || echo "[WARN] Python/SimpleITK check failed"
echo ""

# Step 4: Quick module test
echo "Step 4: Quick module import test..."
echo ""
docker compose run --rm medimages julia --project=. -e '
    using MedImages
    println("MedImages loaded successfully")
    println("Available submodules:")
    println("  - MedImage_data_struct")
    println("  - Load_and_save")
    println("  - Basic_transformations")
    println("  - Spatial_metadata_change")
    println("  - Resample_to_target")
'
echo ""

echo "========================"
echo "Quick start complete!"
echo ""
echo "Next steps:"
echo "  make shell      - Start Julia REPL"
echo "  make test       - Run test suite"
echo "  make benchmark  - Run GPU benchmarks"
echo "  make help       - Show all commands"
