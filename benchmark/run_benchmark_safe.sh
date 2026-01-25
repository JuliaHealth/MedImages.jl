#!/bin/bash
# GPU Benchmark Runner with Memory Management
# Ensures proper GPU memory cleanup before and during benchmark execution

set -e

echo "================================================================================"
echo "GPU Benchmark with Memory Management"
echo "================================================================================"
date

cd /workspaces/MedImages.jl/benchmark

# Step 1: Kill any existing Julia processes
echo ""
echo "Step 1: Terminating existing Julia processes..."
killall -9 julia 2>/dev/null || true
sleep 2
ps aux | grep julia | grep -v grep || echo "No Julia processes found"

# Step 2: Wait for GPU to cool down and memory to free
echo ""
echo "Step 2: Waiting for GPU memory to stabilize..."
sleep 5

# Step 3: Run benchmark with increased memory limits and explicit cleanup
echo ""
echo "Step 3: Starting GPU benchmark..."
echo "  Configuration:"
echo "    - GPU: CUDA_VISIBLE_DEVICES=1 (RTX 3090, 23GB)"
echo "    - Output: gpu1_final_complete/"
echo "    - Timeout: 3600s (1 hour)"
echo "    - Backends: CUDA"
echo "    - Images: Synthetic (xlarge, large, medium)"
echo ""

CUDA_VISIBLE_DEVICES=1 \
  JULIA_NUM_THREADS=4 \
  timeout 3600 \
  julia --project=. \
    -e 'ENV["JULIA_CUDA_MEMORY_POOL"] = "none"' \
    run_gpu_benchmarks.jl \
      --synthetic \
      --backends=cuda \
      --operations=all \
      --output=gpu1_final_complete \
  2>&1 | tee gpu1_final_complete_rerun.log

EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Benchmark completed with exit code: $EXIT_CODE"
date
echo "================================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Benchmark completed successfully!"
    echo "Results saved to:"
    echo "  - gpu1_final_complete/results.csv"
    echo "  - gpu1_final_complete/report.md"
    echo "  - gpu1_final_complete_rerun.log"
else
    echo "✗ Benchmark failed with exit code: $EXIT_CODE"
    echo "Check gpu1_final_complete_rerun.log for details"
fi

exit $EXIT_CODE
