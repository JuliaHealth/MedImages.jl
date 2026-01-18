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

Medical imaging software suffers from fundamental fragmentation:

- **Format proliferation**: NIfTI, DICOM, HDF5, and proprietary formats each require separate tooling
- **Spatial metadata complexity**: Origin, spacing, and direction cosines are handled inconsistently across libraries
- **CPU-only legacy**: Most medical imaging tools predate GPU computing and lack hardware acceleration
- **Non-differentiable operations**: Traditional image processing breaks gradient flow for deep learning integration

Researchers waste time on data wrangling instead of algorithm development.

---

## The Solution

![Concept: The Solution](docs/assets/frame-02.png)

MedImages.jl provides a unified approach to medical image processing:

- **Single data structure**: The `MedImage` struct encapsulates voxel data with complete spatial metadata
- **Format agnostic**: Transparent I/O for NIfTI, DICOM folders, and HDF5 files
- **GPU acceleration**: CUDA support via KernelAbstractions.jl with zero code changes
- **Full differentiability**: Zygote and Enzyme compatible for end-to-end gradient computation
- **Spatial correctness**: All transforms preserve origin, spacing, and orientation metadata

---

## Architecture Overview

![Architecture: Overview](docs/assets/frame-03.png)

```
                        MedImages.jl Architecture
+-------------------------------------------------------------------------+
|                                                                         |
| INPUT FORMATS              CORE STRUCTURE              OUTPUT FORMATS   |
| +----------+              +-------------+              +----------+     |
| | NIfTI    |----+         |  MedImage   |         +---| NIfTI    |     |
| | .nii.gz  |    |    +--->| +---------+ |---+     |   +----------+     |
| +----------+    |    |    | |voxel    | |   |     |                    |
|                 +--->+    | |data     | |   +---->+   +----------+     |
| +----------+    |    |    | |(CPU/GPU)| |   |     +---| HDF5     |     |
| | DICOM    |----+    |    | +---------+ |   |         +----------+     |
| | folder   |         |    | |origin   | |   |                          |
| +----------+         |    | |spacing  | |   |                          |
|                      |    | |direction| |   |                          |
| +----------+         |    | +---------+ |   |                          |
| | HDF5     |---------+    | |metadata | |   |                          |
| +----------+              +------+------+   |                          |
|                                  |          |                          |
+-------------------------------------------------------------------------+
                                   |
              +--------------------+--------------------+
              v                    v                    v
+------------------+   +------------------+   +------------------+
| SPATIAL METADATA |   | BASIC TRANSFORMS |   | RESAMPLING OPS   |
| resample_to_     |   | rotate_mi        |   | resample_to_     |
|   spacing        |   | crop_mi          |   |   image          |
| change_          |   | pad_mi           |   | (registration)   |
|   orientation    |   | translate_mi     |   +------------------+
+------------------+   | scale_mi         |
                       +------------------+
```

---

## MedImage Data Structure

![Architecture: MedImage Data Structure](docs/assets/frame-04.png)

The `MedImage` struct is the central data type:

| Field | Type | Description |
|-------|------|-------------|
| `voxel_data` | `Array` | Multidimensional voxel array (CPU or GPU) |
| `origin` | `Tuple{F64,F64,F64}` | Physical origin coordinates in mm |
| `spacing` | `Tuple{F64,F64,F64}` | Voxel dimensions in mm |
| `direction` | `NTuple{9,F64}` | Direction cosines (3x3 matrix flattened) |
| `image_type` | `Image_type` | Modality: CT, MRI, or PET |
| `image_subtype` | `Image_subtype` | Specific acquisition type |
| `current_device` | `current_device_enum` | CPU or GPU backend indicator |
| `metadata` | `Dict` | Extensible metadata storage |

---

## Type Enumerations

![Implementation: Type Enumerations](docs/assets/frame-05.png)

**Image Type Enum**

| Value | Description |
|-------|-------------|
| `CT` | Computed Tomography |
| `MRI` | Magnetic Resonance Imaging |
| `PET` | Positron Emission Tomography |

**Image Subtype Enum**

| Value | Description |
|-------|-------------|
| `CT_STANDARD` | Standard CT acquisition |
| `MRI_T1` | T1-weighted MRI |
| `MRI_T2` | T2-weighted MRI |
| `MRI_FLAIR` | Fluid-attenuated inversion recovery |
| `PET_FDG` | FDG-PET metabolic imaging |

**Device Enum**

| Value | Description |
|-------|-------------|
| `CPU_en` | Standard CPU arrays |
| `GPU_en` | CUDA GPU arrays via CUDA.jl |

---

## File I/O Operations

![Implementation: File I/O Operations](docs/assets/frame-06.png)

**Loading Images**

```julia
using MedImages

# Load NIfTI file
ct = load_image("scan.nii.gz", "CT")

# Load DICOM folder
mri = load_image("dicom_folder/", "MRI")

# Access spatial metadata
println("Shape: ", size(ct.voxel_data))
println("Origin: ", ct.origin)
println("Spacing: ", ct.spacing)
println("Direction: ", ct.direction)
```

**Saving Images**

```julia
# Export to NIfTI
create_nii_from_medimage(ct, "output.nii.gz")

# Save to HDF5
save_to_hdf5(ct, "output.h5")
```

---

## Spatial Coordinate System

![Concept: Spatial Coordinate System](docs/assets/frame-07.png)

Medical images exist in physical space, not just voxel indices. MedImages.jl maintains three key spatial properties:

**Origin**: The physical coordinates (x, y, z) of the first voxel corner in millimeters. This anchors the image in world space.

**Spacing**: The physical dimensions of each voxel in millimeters. Determines the resolution and scale of the image.

**Direction Cosines**: A 3x3 rotation matrix (stored as 9 values) defining the orientation of voxel axes relative to world axes. This encodes patient orientation.

All transforms in MedImages.jl automatically update these properties to maintain spatial correctness.

---

## Orientation Codes

![Architecture: Orientation Codes](docs/assets/frame-08.png)

Standard orientation codes define anatomical coordinate systems:

| Code | Meaning | Use Case |
|------|---------|----------|
| `ORIENTATION_RAS` | Right-Anterior-Superior | Neuroimaging standard (FreeSurfer, FSL) |
| `ORIENTATION_LPS` | Left-Posterior-Superior | DICOM and ITK standard |
| `ORIENTATION_RAI` | Right-Anterior-Inferior | Alternative neuroimaging |
| `ORIENTATION_LPI` | Left-Posterior-Inferior | Alternative DICOM |
| `ORIENTATION_RPS` | Right-Posterior-Superior | Specialized applications |
| `ORIENTATION_LAS` | Left-Anterior-Superior | Specialized applications |
| `ORIENTATION_RPI` | Right-Posterior-Inferior | Specialized applications |
| `ORIENTATION_LAI` | Left-Anterior-Inferior | Specialized applications |

The first letter indicates left-right axis, second indicates anterior-posterior, third indicates superior-inferior.

---

## Basic Transformations

![Implementation: Basic Transformations](docs/assets/frame-09.png)

All transforms accept an interpolation method and preserve spatial metadata:

```julia
using MedImages

ct = load_image("scan.nii.gz", "CT")

# Rotate 90 degrees around z-axis
rotated = rotate_mi(ct, :z, 90.0, Linear_en)

# Crop to region of interest (start indices, size)
cropped = crop_mi(ct, (50, 50, 20), (100, 100, 60), Linear_en)

# Pad with zeros (beginning padding, end padding, fill value)
padded = pad_mi(ct, (10, 10, 5), (10, 10, 5), 0.0, Linear_en)

# Translate origin by offset along axis
translated = translate_mi(ct, 25.0, :x, Linear_en)

# Scale by factor (affects spacing)
scaled = scale_mi(ct, 2.0, Linear_en)
```

---

## Interpolation Methods

![Architecture: Interpolation Methods](docs/assets/frame-10.png)

Choose interpolation based on your use case:

| Method | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| `Nearest_neighbour_en` | Fast | Discrete | Segmentation masks, label maps |
| `Linear_en` | Medium | Good | General CT/MRI processing |
| `B_spline_en` | Slow | Smooth | High-quality visualization, publication figures |

**Critical**: Always use `Nearest_neighbour_en` for segmentation masks to preserve label integrity. Linear and B-spline interpolation will create invalid intermediate values.

---

## Resampling Operations

![Implementation: Resampling Operations](docs/assets/frame-11.png)

**Change Resolution**

```julia
# Resample to isotropic 1mm spacing
isotropic = resample_to_spacing(ct, (1.0, 1.0, 1.0), Linear_en)

# Resample to specific spacing
resampled = resample_to_spacing(ct, (0.5, 0.5, 2.0), Linear_en)
```

**Change Orientation**

```julia
# Convert to neuroimaging standard
ras = change_orientation(ct, ORIENTATION_RAS)

# Convert to DICOM standard
lps = change_orientation(ct, ORIENTATION_LPS)
```

---

## Cross-Modal Registration

![Example: Cross-Modal Registration](docs/assets/frame-12.png)

Align images from different modalities to a common space:

```julia
# Load CT and PET images
ct = load_image("ct.nii.gz", "CT")
pet = load_image("pet.nii.gz", "PET")

# Resample PET to match CT geometry
# This aligns origin, spacing, and orientation
aligned_pet = resample_to_image(ct, pet, Linear_en)

# Now both images have identical:
# - Voxel grid dimensions
# - Origin coordinates
# - Spacing values
# - Direction cosines

# Save aligned result
create_nii_from_medimage(aligned_pet, "pet_aligned_to_ct.nii.gz")
```

This operation is essential for multi-modal analysis, fusion imaging, and preparing training data for deep learning.

---

## GPU Backend

![Architecture: GPU Backend](docs/assets/frame-13.png)

MedImages.jl uses KernelAbstractions.jl for hardware-agnostic GPU kernels:

```
+-------------------+     +----------------------+     +------------------+
|   User Code       | --> | KernelAbstractions   | --> |   CUDA.jl        |
|   (unchanged)     |     |   (dispatch layer)   |     |   (GPU backend)  |
+-------------------+     +----------------------+     +------------------+
        |                          |                           |
        v                          v                           v
+-------------------+     +----------------------+     +------------------+
|   MedImage with   |     |   Automatic kernel   |     |   Parallel       |
|   CuArray data    |     |   selection          |     |   execution      |
+-------------------+     +----------------------+     +------------------+
```

The same transform functions work on both CPU and GPU data. Backend selection is automatic based on array type.

---

## GPU Usage

![Example: GPU Usage](docs/assets/frame-14.png)

```julia
using MedImages
using CUDA

# Load image on CPU
ct = load_image("scan.nii.gz", "CT")

# Transfer to GPU
gpu_ct = deepcopy(ct)
gpu_ct.voxel_data = CuArray(Float32.(ct.voxel_data))
gpu_ct.current_device = GPU_en

# All operations now run on GPU
resampled = resample_to_spacing(gpu_ct, (2.0, 2.0, 2.0), Linear_en)
rotated = rotate_mi(resampled, :z, 45.0, Linear_en)

# Transfer back to CPU for saving
cpu_result = deepcopy(rotated)
cpu_result.voxel_data = Array(rotated.voxel_data)
cpu_result.current_device = CPU_en

create_nii_from_medimage(cpu_result, "output.nii.gz")
```

---

## Differentiability

![Concept: Differentiability](docs/assets/frame-15.png)

MedImages.jl supports automatic differentiation through two mechanisms:

**Zygote.jl**: Reverse-mode AD via ChainRulesCore rrules. Works with standard Julia arrays and integrates with Flux.jl neural networks.

**Enzyme.jl**: High-performance AD that works with GPU kernels. Required for differentiating through KernelAbstractions code on CUDA.

All resampling and interpolation operations have defined gradients, enabling:

- Spatial transformer networks
- Differentiable rendering
- Image registration with gradient descent
- End-to-end learning with geometric augmentation

---

## Gradient Computation

![Implementation: Gradient Computation](docs/assets/frame-16.png)

```julia
using MedImages
using Zygote

# Define a loss function using MedImages operations
function spatial_loss(voxel_data)
    im = MedImage(
        voxel_data = voxel_data,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        image_type = CT,
        image_subtype = CT_STANDARD,
        current_device = CPU_en,
        metadata = Dict()
    )

    # Differentiable resampling
    resampled = resample_to_spacing(im, (2.0, 2.0, 2.0), Linear_en)

    # Compute scalar loss
    return sum(resampled.voxel_data .^ 2)
end

# Compute gradients
data = rand(Float32, 64, 64, 64)
grads = Zygote.gradient(spatial_loss, data)

# grads[1] contains dL/d(voxel_data)
```

---

## Complete Pipeline

![Example: Complete Pipeline](docs/assets/frame-17.png)

End-to-end example: load, process, and save a medical image:

```julia
using MedImages

# 1. Load input image
ct = load_image("input_ct.nii.gz", "CT")
println("Original shape: ", size(ct.voxel_data))
println("Original spacing: ", ct.spacing)

# 2. Resample to isotropic 1mm resolution
isotropic = resample_to_spacing(ct, (1.0, 1.0, 1.0), Linear_en)
println("Isotropic shape: ", size(isotropic.voxel_data))

# 3. Reorient to standard neuroimaging convention
ras = change_orientation(isotropic, ORIENTATION_RAS)

# 4. Crop to brain region (example coordinates)
cropped = crop_mi(ras, (50, 30, 20), (160, 200, 180), Linear_en)

# 5. Apply rotation for alignment
aligned = rotate_mi(cropped, :z, 5.0, Linear_en)

# 6. Save processed result
create_nii_from_medimage(aligned, "processed_ct.nii.gz")
println("Processing complete")
```

---

## API Quick Reference

![Result: API Quick Reference](docs/assets/frame-18.png)

| Function | Description |
|----------|-------------|
| `load_image(path, type)` | Load NIfTI or DICOM files |
| `create_nii_from_medimage(im, path)` | Export to NIfTI format |
| `save_to_hdf5(im, path)` | Export to HDF5 format |
| `resample_to_spacing(im, spacing, interp)` | Change voxel resolution |
| `change_orientation(im, code)` | Reorient to standard |
| `resample_to_image(fixed, moving, interp)` | Align moving to fixed geometry |
| `rotate_mi(im, axis, angle, interp)` | 3D rotation around axis |
| `crop_mi(im, start, size, interp)` | Crop with origin adjustment |
| `pad_mi(im, beg, end, val, interp)` | Pad with origin adjustment |
| `translate_mi(im, offset, axis, interp)` | Translate origin |
| `scale_mi(im, factor, interp)` | Scale image |

**Interpolation Constants**

- `Nearest_neighbour_en` - Discrete values, fast
- `Linear_en` - Smooth interpolation, balanced
- `B_spline_en` - High quality, slow

**Orientation Constants**

- `ORIENTATION_RAS` - Neuroimaging standard
- `ORIENTATION_LPS` - DICOM standard
- `ORIENTATION_RAI`, `ORIENTATION_LPI`, `ORIENTATION_RPS`, `ORIENTATION_LAS`, `ORIENTATION_RPI`, `ORIENTATION_LAI`

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
