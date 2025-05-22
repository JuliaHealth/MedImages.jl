<div align="center">
  <img src="./docs/src/assets/logo.png" alt="MedImages.jl JuliaHealth Logo" width="200"/>
  <h1>MedImages.jl</h1>
  <p><em>A comprehensive Julia library for standardized 3D and 4D medical imaging data handling</em></p>
</div>

## Overview

MedImages.jl provides a standardized framework for handling 3D and 4D medical imaging data. The metadata structure is loosely based on the BIDS format<sup>[1](#references)</sup>.

This project aims to create a unified approach to medical image processing across various modalities including CT, MRI, and PET scans. Currently, ultrasonography support is not included, and we welcome contributors with expertise in this area.

## Features & Development Roadmap

| Feature Category | Status | Description |
|------------------|--------|-------------|
| **Data Structure Design** | ✅ | Core data structures for medical imaging standardization |
| **Data Loading** | ✅ | Support for common medical imaging formats |
| **Spatial Transformations** | 🚧 | Advanced spatial processing with metadata preservation |
| **Persistence Layer** | 🚧 | Efficient storage and retrieval mechanisms |

### Data Structure Design

The core architecture manages these key components:

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

### Data Loading Capabilities

| Format | Implementation | Status |
|--------|---------------|--------|
| **NIfTI** | via Nifti.jl | ✅ |
| **DICOM** | via Dicom.jl | ✅ |
| **MHA** | direct implementation | 🚧 |

### Spatial Transformations

Our spatial processing framework preserves metadata while enabling:

- **Orientation standardization** to a common reference frame (e.g., RAS)
- **Spacing/resolution adjustment** with appropriate interpolation methods
- **Cross-modality resampling** for multi-modal registration
- **Region-of-interest operations** (cropping, dilation) with origin adjustments

### Persistence Layer

| Feature | Description | Status |
|---------|-------------|--------|
| **HDF5-based storage** | Arrays with metadata attributes | ✅ |
| **Device-agnostic I/O** | Operations for CPU/GPU | 🚧 |
| **Format conversion** | Exporting to standard medical formats | 🚧 |

## Development Status

This project is under active development. The spatial transformation components present the most significant challenges due to numerous edge cases. We're exploring solutions based on packages like [MetaArrays.jl](https://github.com/haberdashPI/MetaArrays.jl).

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

## Contributing

Contributions are welcome! If you have expertise in medical imaging, particularly ultrasonography, or experience with the technical challenges described above, please consider getting involved.

## References

[1] Gorgolewski, K.J., Auer, T., Calhoun, V.D. et al. The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. Sci Data 3, 160044 (2016). https://www.nature.com/articles/sdata201644