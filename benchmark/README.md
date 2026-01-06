# MedImages.jl GPU Benchmark Suite

Comprehensive GPU benchmarking suite for MedImages.jl using large medical imaging datasets from The Cancer Imaging Archive (TCIA).

## Overview

This benchmark suite tests GPU performance for medical image processing operations using NVIDIA CUDA, comparing against CPU baseline and SimpleITK reference implementation.

**Key Features:**
- Downloads large NIFTI images from TCIA (512x512x256+ and 1024x1024x512)
- Benchmarks all major MedImages.jl GPU operations
- Compares CPU vs CUDA performance
- Generates comprehensive performance reports

## Installation

1. Activate the benchmark environment:
```bash
cd benchmark
julia --project=.
```

2. Install dependencies:
```julia
using Pkg
Pkg.instantiate()
```

3. Verify CUDA is available:
```julia
using CUDA
CUDA.functional()  # Should return true
```

## Usage

### Step 1: Download Data from TCIA

Download large medical imaging datasets:

```bash
julia --project=.. download_tcia_data.jl
```

This will:
- Query TCIA for large CT/MRI scans (>300 slices)
- Download top 5 largest series
- Save to `benchmark_data/dicom_raw/`
- Generate `benchmark_data/metadata.json`

**Note:** Downloads can be several GB. Ensure sufficient disk space.

### Step 2: Convert DICOM to NIFTI

Convert downloaded DICOM files to NIFTI format:

```bash
julia --project=.. convert_dicom_to_nifti.jl
```

This will:
- Convert all DICOM series to NIFTI
- Categorize by size (large/xlarge)
- Save to `benchmark_data/nifti/{large,xlarge}/`
- Generate `benchmark_data/nifti/benchmark_catalog.csv`

### Step 3: Run Benchmarks

Run the full benchmark suite:

```bash
julia --project=.. run_gpu_benchmarks.jl
```

**Options:**
- `--synthetic`: Use synthetic test images (no data download needed)
- `--operations OPS`: Comma-separated operations to benchmark
  - `interpolate`: Direct interpolation kernel
  - `resample`: Spacing resampling
  - `cross_resample`: Cross-image registration
  - `rotate`: Rotation transformations
  - `crop`: Crop operations
  - `pad`: Padding operations
  - `orientation`: Orientation changes
  - `all`: All operations (default)
- `--backends BACK`: Backends to test (`cpu`, `cuda`, `all`)
- `--catalog FILE`: Path to benchmark catalog CSV
- `--output DIR`: Output directory (default: `benchmark_results/`)

**Examples:**

```bash
# Run all benchmarks on real data
julia --project=.. run_gpu_benchmarks.jl

# Quick test with synthetic data
julia --project=.. run_gpu_benchmarks.jl --synthetic

# Benchmark only interpolation and resampling
julia --project=.. run_gpu_benchmarks.jl --operations interpolate,resample

# CPU only (no GPU)
julia --project=.. run_gpu_benchmarks.jl --backends cpu
```

## Output

Results are saved to `benchmark_results/`:
- `results.csv`: Raw benchmark data
- `report.md`: Comprehensive markdown report with:
  - System information
  - Performance tables
  - CPU vs CUDA speedup comparisons
  - Memory transfer statistics
  - Optimization recommendations

## Benchmark Operations

The suite benchmarks the following operations:

### 1. Interpolation Kernel
Direct GPU kernel performance with varying point counts (100K, 1M, 10M points).
- Nearest neighbor interpolation
- Trilinear interpolation

### 2. Resampling
Spacing changes with various resampling factors.
- Upsampling: 2x, 4x
- Downsampling: 0.5x, 0.25x
- Isotropic and anisotropic spacing

### 3. Cross-Image Resampling
Register one image to another's coordinate system.
- Different orientation codes (LAS, RAS, RPI)
- Complete pipeline: orientation + interpolation

### 4. Rotation
Rotate images by various angles.
- 90, 180, 270 degrees
- Around Z-axis

### 5. Crop/Pad
Array slicing and padding operations.
- Small (50%), medium (30%), large (10%) ratios

### 6. Orientation Changes
Transform between different medical imaging orientations.
- LAS, RAS, RPI, LPS

### 7. Memory Transfer
CPU <-> GPU transfer performance.
- Transfer rates (MB/s)
- PCIe bandwidth analysis

## Expected Performance

For large images (512³):
- **Interpolation:** 10-50x GPU speedup
- **Resampling:** 20-100x GPU speedup
- **Rotation:** CPU-bound (current implementation)
- **Memory transfer:** ~10-24 GB/s (PCIe Gen3/4)

For xlarge images (1024³):
- **Interpolation:** 20-100x GPU speedup
- **Memory-bound operations:** Higher speedup ratios

## Configuration

Edit `benchmark_config.jl` to customize:
- Image sizes to test
- Interpolation methods
- Spacing configurations
- Resampling factors
- Number of benchmark samples
- Output paths

## Troubleshooting

### CUDA Not Available
- Ensure CUDA.jl is properly installed: `] add CUDA`
- Check NVIDIA drivers: `nvidia-smi`
- Verify GPU compatibility: `CUDA.functional()`

### Out of Memory
- Reduce image sizes in config
- Use `--operations` to run subset of benchmarks
- Close other GPU-consuming applications

### Download Failures
- Check internet connection
- TCIA may be temporarily unavailable
- Try different collections: edit `download_tcia_data.jl`

### Conversion Failures
- Verify DICOM files are valid
- Check ITKIOWrapper installation
- Try with `--synthetic` flag

## Architecture

**Files:**
- `download_tcia_data.jl`: TCIA API client
- `convert_dicom_to_nifti.jl`: DICOM→NIFTI converter
- `benchmark_config.jl`: Configuration parameters
- `gpu_benchmarks.jl`: Core benchmark functions
- `benchmark_utils.jl`: Reporting and visualization
- `run_gpu_benchmarks.jl`: Main benchmark orchestrator
- `Project.toml`: Dependencies

**Data Flow:**
```
TCIA API → DICOM (zip) → Extract → Convert → NIFTI → Catalog → Benchmark → Report
```

## Contributing

To add new benchmarks:
1. Add benchmark function to `gpu_benchmarks.jl`
2. Add operation config to `benchmark_config.jl`
3. Integrate in `run_gpu_benchmarks.jl`
4. Update this README

## License

Same as MedImages.jl

## Acknowledgments

- The Cancer Imaging Archive (TCIA) for providing public medical imaging data
- NIH/NCI for TCIA support
- Contributors to MedImages.jl
