# MedImages.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliahealth.org/MedImages.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliahealth.org/MedImages.jl/dev)
[![Build Status](https://github.com/JuliaHealth/MedImages.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaHealth/MedImages.jl/actions/workflows/CI.yml)

A comprehensive Julia library for GPU-accelerated, differentiable medical image processing.

---

## Table of Contents

<div align="center">

[![The Challenge](https://img.shields.io/badge/The_Challenge-orange)](#the-challenge)
[![The Solution](https://img.shields.io/badge/The_Solution-orange)](#the-solution)
[![Architecture](https://img.shields.io/badge/Architecture-orange)](#architecture-overview)
[![Data Structure](https://img.shields.io/badge/Data_Structure-orange)](#medimage-data-structure)
[![Type Enums](https://img.shields.io/badge/Type_Enums-orange)](#type-enumerations)
[![File I/O](https://img.shields.io/badge/File_I%2FO-orange)](#file-io-operations)
[![Spatial System](https://img.shields.io/badge/Spatial_System-orange)](#spatial-coordinate-system)
[![Orientations](https://img.shields.io/badge/Orientations-orange)](#orientation-codes)
[![Transforms](https://img.shields.io/badge/Transforms-orange)](#basic-transformations)
[![Interpolation](https://img.shields.io/badge/Interpolation-orange)](#interpolation-methods)
[![Resampling](https://img.shields.io/badge/Resampling-orange)](#resampling-operations)
[![Registration](https://img.shields.io/badge/Registration-orange)](#cross-modal-registration)
[![GPU Backend](https://img.shields.io/badge/GPU_Backend-orange)](#gpu-backend)
[![GPU Usage](https://img.shields.io/badge/GPU_Usage-orange)](#gpu-usage)
[![Differentiability](https://img.shields.io/badge/Differentiability-orange)](#differentiability)
[![Gradients](https://img.shields.io/badge/Gradients-orange)](#gradient-computation)
[![Pipeline](https://img.shields.io/badge/Pipeline-orange)](#complete-pipeline)
[![API Reference](https://img.shields.io/badge/API_Reference-orange)](#api-quick-reference)
[![Docker](https://img.shields.io/badge/Docker-gray)](#quick-start-with-docker)
[![Contributing](https://img.shields.io/badge/Contributing-gray)](#contributing)
[![References](https://img.shields.io/badge/References-gray)](#references)

</div>

---

## The Challenge

![Concept: The Challenge](docs/assets/frame-01.png)

---

## The Solution

![Concept: The Solution](docs/assets/frame-02.png)

---

## Architecture Overview

![Architecture: Overview](docs/assets/frame-03.png)

---

## MedImage Data Structure

![Architecture: MedImage Data Structure](docs/assets/frame-04.png)

---

## Type Enumerations

![Implementation: Type Enumerations](docs/assets/frame-05.png)

---

## File I/O Operations

![Implementation: File I/O Operations](docs/assets/frame-06.png)

---

## Spatial Coordinate System

![Concept: Spatial Coordinate System](docs/assets/frame-07.png)

---

## Orientation Codes

![Architecture: Orientation Codes](docs/assets/frame-08.png)

---

## Basic Transformations

![Implementation: Basic Transformations](docs/assets/frame-09.png)

---

## Interpolation Methods

![Architecture: Interpolation Methods](docs/assets/frame-10.png)

---

## Resampling Operations

![Implementation: Resampling Operations](docs/assets/frame-11.png)

---

## Cross-Modal Registration

![Example: Cross-Modal Registration](docs/assets/frame-12.png)

---

## GPU Backend

![Architecture: GPU Backend](docs/assets/frame-13.png)

---

## GPU Usage

![Example: GPU Usage](docs/assets/frame-14.png)

---

## Differentiability

![Concept: Differentiability](docs/assets/frame-15.png)

---

## Gradient Computation

![Implementation: Gradient Computation](docs/assets/frame-16.png)

---

## Complete Pipeline

![Example: Complete Pipeline](docs/assets/frame-17.png)

---

## API Quick Reference

![Result: API Quick Reference](docs/assets/frame-18.png)

---

## Quick Start with Docker

The easiest way to get started is using Docker with GPU support for benchmarks.

### Prerequisites

- Docker with NVIDIA GPU support (for GPU benchmarks)
- Or Docker without GPU (CPU-only mode available)

### Build and Run

```bash
# Build the Docker image
make build

# Start interactive Julia REPL (with GPU)
make shell

# Start interactive Julia REPL (CPU only)
make shell-cpu
```

### Run Tests

```bash
# Run the full test suite
make test

# Run tests in CPU-only mode
make test-cpu
```

### Run Benchmarks

```bash
# Run GPU benchmarks (uses synthetic data)
make benchmark

# Run CPU-only benchmarks
make benchmark-cpu

# Custom benchmark options
make benchmark-custom ARGS="--size 64 --iterations 5"
```

### Verify Setup

```bash
# Check CUDA/GPU availability
make check-cuda

# Check Python/SimpleITK setup
make check-python

# Run quick start verification
./scripts/quick-start.sh
```

### Test Data

Test data files are expected in `test_data/`:

- `volume-0.nii.gz` - Primary NIfTI test file
- `synthethic_small.nii.gz` - Synthetic test file
- `ScalarVolume_0/` - DICOM test directory

```bash
# Check test data availability
./scripts/check-test-data.sh

# Download benchmark data from TCIA
make download-data

# Convert DICOM to NIfTI for benchmarks
make convert-data
```

Note: Benchmarks use synthetic data by default (`make benchmark`). Real data download is only needed for `make benchmark-full`.

### All Make Commands

```bash
make help  # Show all available commands
```

---

## Contributing

Contributions are welcome! If you have expertise in medical imaging, particularly ultrasonography, or experience with the technical challenges described above, please consider getting involved.

---

## References

[1] Gorgolewski, K.J., Auer, T., Calhoun, V.D. et al. The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. Sci Data 3, 160044 (2016). https://www.nature.com/articles/sdata201644
