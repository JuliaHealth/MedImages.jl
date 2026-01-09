# MedImages.jl Docker Makefile
# Usage: make <target>
#
# Quick Start:
#   make build     - Build Docker image
#   make shell     - Interactive Julia REPL with GPU
#   make test      - Run test suite
#   make benchmark - Run GPU benchmarks

.PHONY: build build-no-cache shell shell-cpu test test-cpu benchmark benchmark-cpu \
        benchmark-custom benchmark-simpleitk benchmark-shell check-cuda check-python \
        clean clean-all logs help

# Docker image configuration
IMAGE_NAME := medimages
IMAGE_TAG := latest

# Default target
.DEFAULT_GOAL := help

#-----------------------------------------------------------------------------
# Build Targets
#-----------------------------------------------------------------------------

## Build the Docker image
build:
	docker compose build medimages

## Build without cache (clean rebuild)
build-no-cache:
	docker compose build --no-cache medimages

#-----------------------------------------------------------------------------
# Interactive Targets
#-----------------------------------------------------------------------------

## Start interactive Julia REPL with GPU support
shell:
	docker compose run --rm medimages

## Start interactive Julia REPL (CPU only, no GPU required)
shell-cpu:
	docker compose run --rm medimages-cpu

## Open Julia REPL in benchmark project
benchmark-shell:
	docker compose run --rm -w /workspace/MedImages.jl/benchmark benchmark julia --project=.

#-----------------------------------------------------------------------------
# Test Targets
#-----------------------------------------------------------------------------

## Run the test suite with GPU support
test:
	docker compose run --rm test

## Run the test suite (CPU only)
test-cpu:
	docker compose run --rm test-cpu

#-----------------------------------------------------------------------------
# Benchmark Targets
#-----------------------------------------------------------------------------

## Run GPU benchmarks with synthetic data
benchmark:
	docker compose run --rm benchmark

## Run CPU-only benchmarks
benchmark-cpu:
	docker compose run --rm -e CUDA_VISIBLE_DEVICES="" benchmark \
		julia --project=. run_gpu_benchmarks.jl --synthetic --backends cpu

## Run benchmarks with custom options
## Usage: make benchmark-custom ARGS="--operations resample --backends cuda"
benchmark-custom:
	docker compose run --rm benchmark julia --project=. run_gpu_benchmarks.jl $(ARGS)

## Run SimpleITK comparison benchmarks
benchmark-simpleitk:
	docker compose run --rm -w /workspace/MedImages.jl/benchmark medimages \
		julia --project=. simpleitk_benchmarks.jl

## Run full benchmark suite with real data (requires downloaded data)
benchmark-full:
	docker compose run --rm benchmark \
		julia --project=. run_gpu_benchmarks.jl --backends all

#-----------------------------------------------------------------------------
# Verification Targets
#-----------------------------------------------------------------------------

## Check CUDA/GPU availability
check-cuda:
	@echo "Checking CUDA availability..."
	@docker compose run --rm medimages julia --project=. -e '\
		using CUDA; \
		println("CUDA functional: ", CUDA.functional()); \
		if CUDA.functional() \
			println("GPU: ", CUDA.name(CUDA.device())); \
			println("Memory: ", round(CUDA.totalmem(CUDA.device()) / 1e9, digits=2), " GB"); \
			println("CUDA version: ", CUDA.runtime_version()); \
		end'

## Check Python/SimpleITK availability
check-python:
	@echo "Checking Python/SimpleITK availability..."
	@docker compose run --rm medimages julia --project=. -e '\
		using PyCall; \
		sitk = pyimport("SimpleITK"); \
		np = pyimport("numpy"); \
		println("SimpleITK version: ", sitk.Version()); \
		println("NumPy version: ", np.__version__)'

## Run all verification checks
check-all: check-cuda check-python
	@echo "All checks complete."

#-----------------------------------------------------------------------------
# Cleanup Targets
#-----------------------------------------------------------------------------

## Clean up Docker containers and volumes
clean:
	docker compose down -v --rmi local
	-docker volume rm medimages-julia-depot 2>/dev/null || true

## Remove all build artifacts and start fresh
clean-all: clean
	-docker rmi -f $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	rm -rf benchmark/benchmark_results/

#-----------------------------------------------------------------------------
# Utility Targets
#-----------------------------------------------------------------------------

## Show logs from containers
logs:
	docker compose logs

## Download benchmark test data (TCIA)
download-data:
	docker compose run --rm -w /workspace/MedImages.jl/benchmark medimages \
		julia --project=. download_tcia_data.jl

## Convert DICOM to NIfTI for benchmarks
convert-data:
	docker compose run --rm -w /workspace/MedImages.jl/benchmark medimages \
		julia --project=. convert_dicom_to_nifti.jl

#-----------------------------------------------------------------------------
# Help
#-----------------------------------------------------------------------------

## Show this help message
help:
	@echo "MedImages.jl Docker Commands"
	@echo "============================"
	@echo ""
	@echo "Build:"
	@echo "  make build              Build Docker image"
	@echo "  make build-no-cache     Build without cache (clean rebuild)"
	@echo ""
	@echo "Interactive:"
	@echo "  make shell              Julia REPL with GPU"
	@echo "  make shell-cpu          Julia REPL without GPU"
	@echo "  make benchmark-shell    Julia REPL in benchmark project"
	@echo ""
	@echo "Testing:"
	@echo "  make test               Run test suite with GPU"
	@echo "  make test-cpu           Run test suite without GPU"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make benchmark          Run GPU benchmarks (synthetic data)"
	@echo "  make benchmark-cpu      Run CPU-only benchmarks"
	@echo "  make benchmark-full     Run with real data (requires download)"
	@echo "  make benchmark-simpleitk  Run SimpleITK comparison"
	@echo "  make benchmark-custom ARGS='...'  Custom benchmark options"
	@echo ""
	@echo "Verification:"
	@echo "  make check-cuda         Verify CUDA/GPU setup"
	@echo "  make check-python       Verify Python/SimpleITK"
	@echo "  make check-all          Run all verification checks"
	@echo ""
	@echo "Data Management:"
	@echo "  make download-data      Download TCIA benchmark data"
	@echo "  make convert-data       Convert DICOM to NIfTI"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              Remove containers and volumes"
	@echo "  make clean-all          Full cleanup including images"
	@echo ""
	@echo "Example Usage:"
	@echo "  make build && make check-cuda && make test"
	@echo "  make benchmark-custom ARGS='--operations resample,rotate --backends cuda'"
