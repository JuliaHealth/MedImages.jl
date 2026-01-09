<div align="center">
  <img src="./docs/src/assets/logo.png" alt="MedImages.jl JuliaHealth Logo" width="200" align="left" style="margin-right: 20px"/>
  <h1>MedImages.jl</h1>
  <p><em>A comprehensive Julia library for standardized 3D and 4D medical imaging data handling</em></p>
</div>

<br clear="all"/>

## Overview

MedImages.jl provides a standardized framework for handling 3D and 4D medical imaging data. The metadata structure is loosely based on the BIDS format<sup>[1](#references)</sup>.

This project aims to create a unified approach to medical image processing across various modalities including CT, MRI, and PET scans. Currently, ultrasonography support is not included, and we welcome contributors with expertise in this area.

## Features & Development Roadmap

<div align="center">

| Feature Category | Status | Description |
|------------------|:------:|-------------|
| **Data Structure Design** | ✅ | Core data structures for medical imaging standardization |
| **Data Loading** | ✅ | Support for common medical imaging formats |
| **Spatial Transformations** | 🚧 | Advanced spatial processing with metadata preservation |
| **Persistence Layer** | 🚧 | Efficient storage and retrieval mechanisms |

</div>

### Data Structure Design

The core architecture manages these key components:

<div align="center">

<table>
  <tr>
    <th>Component</th>
    <th>Includes</th>
  </tr>
  <tr>
    <td><strong>Voxel data</strong></td>
    <td>Multidimensional arrays</td>
  </tr>
  <tr>
    <td><strong>Spatial metadata</strong></td>
    <td>
      • Origin coordinates<br>
      • Orientation information<br>
      • Spacing/resolution data
    </td>
  </tr>
  <tr>
    <td><strong>Image classification</strong></td>
    <td>
      • Primary type (CT/MRI/PET/label maps)<br>
      • Subtype (e.g., MRI: ADC/DWI/T2)<br>
      • Voxel data type (e.g., Float32)
    </td>
  </tr>
  <tr>
    <td><strong>Study metadata</strong></td>
    <td>
      • Acquisition date/time<br>
      • Patient identifiers<br>
      • Study/Series UIDs<br>
      • Study descriptions<br>
      • Original filenames
    </td>
  </tr>
  <tr>
    <td><strong>Display properties</strong></td>
    <td>
      • Color mappings for labels<br>
      • Window/level values for CT scans
    </td>
  </tr>
  <tr>
    <td><strong>Clinical data</strong></td>
    <td>
      • Patient demographics<br>
      • Contrast administration status
    </td>
  </tr>
  <tr>
    <td><strong>Additional metadata</strong></td>
    <td>Stored in extensible dictionaries</td>
  </tr>
</table>

</div>

### Data Loading Capabilities

<div align="center">

| Format | Implementation | Status |
|--------|---------------|:------:|
| **NIfTI** | via Nifti.jl | ✅ |
| **DICOM** | via Dicom.jl | ✅ |
| **MHA** | direct implementation | 🚧 |

</div>

### Spatial Transformations

Our spatial processing framework preserves metadata while enabling:

- **Orientation standardization** to a common reference frame (e.g., RAS)
- **Spacing/resolution adjustment** with appropriate interpolation methods
- **Cross-modality resampling** for multi-modal registration
- **Region-of-interest operations** (cropping, dilation) with origin adjustments

### Persistence Layer

<div align="center">

| Feature | Description | Status |
|---------|-------------|:------:|
| **HDF5-based storage** | Arrays with metadata attributes | ✅ |
| **Device-agnostic I/O** | Operations for CPU/GPU | 🚧 |
| **Format conversion** | Exporting to standard medical formats | 🚧 |

</div>

## Development Status

This project is under active development. The spatial transformation components present the most significant challenges due to numerous edge cases. We're exploring solutions based on packages like [MetaArrays.jl](https://github.com/haberdashPI/MetaArrays.jl).

<div align="center">

<table>
  <tr>
    <th>Component</th>
    <th>Status</th>
    <th>Priority</th>
  </tr>
  <tr>
    <td>Core data structures</td>
    <td>✅ Complete</td>
    <td>High</td>
  </tr>
  <tr>
    <td>Format loading/saving</td>
    <td>✅ Complete</td>
    <td>High</td>
  </tr>
  <tr>
    <td>Spatial transformations</td>
    <td>🚧 In progress</td>
    <td>High</td>
  </tr>
  <tr>
    <td>GPU compatibility</td>
    <td>🚧 In progress</td>
    <td>Medium</td>
  </tr>
  <tr>
    <td>Ultrasonography support</td>
    <td>📋 Planned</td>
    <td>Low</td>
  </tr>
</table>

</div>

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

## Contributing

Contributions are welcome! If you have expertise in medical imaging, particularly ultrasonography, or experience with the technical challenges described above, please consider getting involved.

## References

[1] Gorgolewski, K.J., Auer, T., Calhoun, V.D. et al. The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. Sci Data 3, 160044 (2016). https://www.nature.com/articles/sdata201644