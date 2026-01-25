#!/bin/bash
set -e
cd /workspaces/MedImages.jl/benchmark
export CUDA_VISIBLE_DEVICES=1
timeout 1800 julia --project=. run_gpu_benchmarks.jl \
  --synthetic \
  --backends=cuda \
  --operations=all \
  --output=gpu1_final_complete
