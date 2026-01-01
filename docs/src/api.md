# API Reference

This page provides comprehensive documentation for all public functions, types, and constants in MedImages.jl.

## Quick Navigation

- [Core Data Types](#core-data-types)
- [I/O Functions](#io-functions)
- [Basic Transformations](#basic-transformations)
- [Spatial Metadata Operations](#spatial-metadata-operations)
- [Resampling Functions](#resampling-functions)
- [HDF5 Storage](#hdf5-storage)
- [Enumerations](#enumerations)
- [Orientation Utilities](#orientation-utilities)

---

## Core Data Types

### MedImage

The primary data structure for representing medical images with comprehensive metadata.

```julia
mutable struct MedImage
    voxel_data                                   # Multidimensional array of image data
    origin::Tuple{Float64,Float64,Float64}       # Physical origin coordinates (x,y,z) in mm
    spacing::Tuple{Float64,Float64,Float64}      # Voxel spacing (x,y,z) in mm
    direction::NTuple{9,Float64}                 # Orientation cosines (3x3 flattened)
    image_type::Image_type                       # CT_type, PET_type, or MRI_type
    image_subtype::Image_subtype                 # Specific imaging protocol
    date_of_saving::DateTime                     # When image was saved
    acquistion_time::DateTime                    # When image was acquired
    patient_id::String                           # Patient identifier
    current_device::current_device_enum          # Processing device (CPU/GPU)
    study_uid::String                            # DICOM study UID
    patient_uid::String                          # Patient UID
    series_uid::String                           # Series UID
    study_description::String                    # Study description
    legacy_file_name::String                     # Original filename
    display_data::Dict{Any,Any}                  # Visualization parameters
    clinical_data::Dict{Any,Any}                 # Patient clinical info
    is_contrast_administered::Bool               # Contrast agent flag
    metadata::Dict{Any,Any}                      # Additional metadata
end
```

#### Fields Description

| Field | Type | Description |
|-------|------|-------------|
| `voxel_data` | `Array` | 3D or 4D array of voxel intensities. For 3D images, dimensions are (Z, Y, X) in Julia array indexing. |
| `origin` | `Tuple{Float64,Float64,Float64}` | Physical coordinates (x,y,z) of the first voxel in millimeters. |
| `spacing` | `Tuple{Float64,Float64,Float64}` | Distance between voxel centers (x,y,z) in millimeters. |
| `direction` | `NTuple{9,Float64}` | Direction cosine matrix flattened row-major. Defines the orientation of the image axes in physical space. |
| `image_type` | `Image_type` | Modality: `CT_type`, `PET_type`, or `MRI_type`. |
| `image_subtype` | `Image_subtype` | Specific protocol: `CT_subtype`, `T1_subtype`, `T2_subtype`, `FLAIR_subtype`, `ADC_subtype`, `DWI_subtype`, `FDG_subtype`, `PSMA_subtype`. |
| `current_device` | `current_device_enum` | Compute device: `CPU_current_device`, `CUDA_current_device`, `AMD_current_device`, `ONEAPI_current_device`. |

---

## I/O Functions

### load_image

```julia
load_image(path::String, type::String)::MedImage
```

Load a medical image from a NIfTI file.

**Arguments:**
- `path::String`: Path to the NIfTI file (.nii or .nii.gz)
- `type::String`: Image modality, either `"CT"` or `"PET"`

**Returns:**
- `MedImage`: Loaded image with spatial metadata

**Example:**
```julia
using MedImages

# Load a CT scan
ct_image = load_image("/path/to/scan.nii.gz", "CT")

# Load a PET scan
pet_image = load_image("/path/to/pet.nii.gz", "PET")
```

**Notes:**
- Uses ITKIOWrapper internally, which defaults to LPS orientation
- Voxel data is stored in Julia array order (dim1=Z, dim2=Y, dim3=X)

---

### create_nii_from_medimage

```julia
create_nii_from_medimage(med_image::MedImage, file_path::String)
```

Save a MedImage object as a NIfTI file.

**Arguments:**
- `med_image::MedImage`: The image to save
- `file_path::String`: Output path (without extension; .nii.gz will be appended)

**Example:**
```julia
# Save processed image
create_nii_from_medimage(processed_image, "/output/processed")
# Creates: /output/processed.nii.gz
```

---

### update_voxel_data

```julia
update_voxel_data(old_image, new_voxel_data::AbstractArray)
```

Create a new MedImage with updated voxel data while preserving all spatial metadata.

**Arguments:**
- `old_image`: Source MedImage for metadata
- `new_voxel_data::AbstractArray`: New voxel data array

**Returns:**
- `MedImage`: New image with updated voxel data

**Example:**
```julia
# Apply custom processing
processed_data = my_filter(image.voxel_data)
new_image = update_voxel_data(image, processed_data)
```

---

### update_voxel_and_spatial_data

```julia
update_voxel_and_spatial_data(old_image, new_voxel_data::AbstractArray,
                               new_origin, new_spacing, new_direction)
```

Create a new MedImage with updated voxel data and spatial metadata.

**Arguments:**
- `old_image`: Source MedImage for non-spatial metadata
- `new_voxel_data::AbstractArray`: New voxel data array
- `new_origin`: New origin tuple (x,y,z)
- `new_spacing`: New spacing tuple (x,y,z)
- `new_direction`: New direction tuple (9 elements)

**Returns:**
- `MedImage`: New image with updated data and spatial metadata

---

## Basic Transformations

### rotate_mi

```julia
rotate_mi(image::MedImage, axis::Int, angle::Float64,
          Interpolator::Interpolator_enum, crop::Bool=true)::MedImage
```

Rotate a medical image around a specified axis.

**Arguments:**
- `image::MedImage`: Input image to rotate
- `axis::Int`: Rotation axis (1, 2, or 3)
- `angle::Float64`: Rotation angle in degrees
- `Interpolator::Interpolator_enum`: Interpolation method
- `crop::Bool=true`: If true, crop result to original dimensions

**Returns:**
- `MedImage`: Rotated image

**Example:**
```julia
# Rotate 45 degrees around axis 3 (typically axial plane)
rotated = rotate_mi(image, 3, 45.0, Linear_en)

# Rotate without cropping to original size
rotated_full = rotate_mi(image, 2, 30.0, Linear_en, false)
```

**Notes:**
- Rotation center is the image center
- Uses Rodrigues' rotation formula internally
- Fill value for areas outside original image is 0.0

---

### crop_mi

```julia
crop_mi(im::MedImage, crop_beg::Tuple{Int64,Int64,Int64},
        crop_size::Tuple{Int64,Int64,Int64},
        Interpolator::Interpolator_enum)::MedImage
```

Crop a region from a medical image.

**Arguments:**
- `im::MedImage`: Input image to crop
- `crop_beg::Tuple{Int64,Int64,Int64}`: Starting indices (0-based) in Julia array order (dim1, dim2, dim3)
- `crop_size::Tuple{Int64,Int64,Int64}`: Size of region to crop in each dimension
- `Interpolator::Interpolator_enum`: Interpolation method (not used in current implementation)

**Returns:**
- `MedImage`: Cropped image with adjusted origin

**Example:**
```julia
# Crop a 100x100x50 region starting at (10, 20, 30)
cropped = crop_mi(image, (10, 20, 30), (100, 100, 50), Linear_en)
```

**Notes:**
- Indices are 0-based for SimpleITK compatibility
- Origin is automatically adjusted to reflect the crop location
- Tuple order is Julia array dimension order: (dim1, dim2, dim3) which maps to (Z, Y, X) physically

---

### pad_mi

```julia
pad_mi(im::MedImage, pad_beg::Tuple{Int64,Int64,Int64},
       pad_end::Tuple{Int64,Int64,Int64}, pad_val,
       Interpolator::Interpolator_enum)::MedImage
```

Pad a medical image with a constant value.

**Arguments:**
- `im::MedImage`: Input image to pad
- `pad_beg::Tuple{Int64,Int64,Int64}`: Voxels to add at the beginning of each axis
- `pad_end::Tuple{Int64,Int64,Int64}`: Voxels to add at the end of each axis
- `pad_val`: Padding value
- `Interpolator::Interpolator_enum`: Interpolation method (not used in current implementation)

**Returns:**
- `MedImage`: Padded image with adjusted origin

**Example:**
```julia
# Pad 10 voxels on each side with value -1000 (air in CT)
padded = pad_mi(image, (10, 10, 10), (10, 10, 10), -1000.0, Linear_en)
```

**Notes:**
- Origin is adjusted to account for padding at the beginning
- Tuple order follows Julia array indexing

---

### translate_mi

```julia
translate_mi(im::MedImage, translate_by::Int64, translate_in_axis::Int64,
             Interpolator::Interpolator_enum)::MedImage
```

Translate a medical image by modifying its origin (metadata-only operation).

**Arguments:**
- `im::MedImage`: Input image
- `translate_by::Int64`: Number of voxels to translate
- `translate_in_axis::Int64`: Axis to translate along (1, 2, or 3)
- `Interpolator::Interpolator_enum`: Interpolation method (not used)

**Returns:**
- `MedImage`: Translated image with modified origin

**Example:**
```julia
# Translate 5 voxels along axis 1 (X in physical space)
translated = translate_mi(image, 5, 1, Linear_en)
```

**Notes:**
- This is a metadata-only operation; the voxel array is unchanged
- Translation distance in mm = translate_by * spacing[axis]

---

### scale_mi

```julia
scale_mi(im::MedImage, scale::Union{Float64, Tuple{Float64,Float64,Float64}},
         Interpolator::Interpolator_enum)::MedImage
```

Scale a medical image, changing its array dimensions.

**Arguments:**
- `im::MedImage`: Input image
- `scale`: Scaling factor (single value for uniform, or tuple for per-axis)
- `Interpolator::Interpolator_enum`: Interpolation method

**Returns:**
- `MedImage`: Scaled image with new dimensions

**Example:**
```julia
# Uniform downscale by half
downscaled = scale_mi(image, 0.5, Linear_en)

# Non-uniform scaling
scaled = scale_mi(image, (1.0, 0.5, 0.5), Linear_en)
```

**Notes:**
- Unlike SimpleITK which keeps output dimensions fixed, MedImages.scale_mi changes array dimensions
- Physical extent is maintained; the number of voxels changes

---

## Spatial Metadata Operations

### resample_to_spacing

```julia
resample_to_spacing(im, new_spacing::Tuple{Float64,Float64,Float64},
                    interpolator_enum, use_cuda=false)::MedImage
```

Resample an image to new voxel spacing.

**Arguments:**
- `im`: Input MedImage
- `new_spacing::Tuple{Float64,Float64,Float64}`: Desired spacing (x,y,z) in mm
- `interpolator_enum`: Interpolation method
- `use_cuda::Bool=false`: Whether to use CUDA acceleration

**Returns:**
- `MedImage`: Resampled image with new spacing and dimensions

**Example:**
```julia
# Resample to isotropic 1mm spacing
isotropic = resample_to_spacing(image, (1.0, 1.0, 1.0), Linear_en)
```

**Notes:**
- Array dimensions are recalculated to maintain physical extent
- Origin is preserved

---

### change_orientation

```julia
change_orientation(im::MedImage, new_orientation::Orientation_code)::MedImage
```

Change the orientation of a medical image.

**Arguments:**
- `im::MedImage`: Input image
- `new_orientation::Orientation_code`: Target orientation

**Returns:**
- `MedImage`: Reoriented image

**Example:**
```julia
# Convert to RAS orientation (common for neuroimaging)
ras_image = change_orientation(image, ORIENTATION_RAS)

# Convert to LPS orientation (common for CT/PET)
lps_image = change_orientation(image, ORIENTATION_LPS)
```

**Supported Orientations:**
- `ORIENTATION_RPI` - Right-Posterior-Inferior
- `ORIENTATION_LPI` - Left-Posterior-Inferior
- `ORIENTATION_RAI` - Right-Anterior-Inferior
- `ORIENTATION_LAI` - Left-Anterior-Inferior
- `ORIENTATION_RPS` - Right-Posterior-Superior
- `ORIENTATION_LPS` - Left-Posterior-Superior
- `ORIENTATION_RAS` - Right-Anterior-Superior
- `ORIENTATION_LAS` - Left-Anterior-Superior

---

## Resampling Functions

### resample_to_image

```julia
resample_to_image(im_fixed::MedImage, im_moving::MedImage,
                  interpolator_enum::Interpolator_enum,
                  value_to_extrapolate=Nothing)::MedImage
```

Resample a moving image to match a fixed image's geometry.

**Arguments:**
- `im_fixed::MedImage`: Reference image defining target geometry
- `im_moving::MedImage`: Image to be resampled
- `interpolator_enum::Interpolator_enum`: Interpolation method
- `value_to_extrapolate`: Value for regions outside moving image (default: median of fixed image corners)

**Returns:**
- `MedImage`: Moving image resampled to fixed image space

**Example:**
```julia
# Register PET to CT space
pet_in_ct_space = resample_to_image(ct_image, pet_image, Linear_en)

# With custom extrapolation value
registered = resample_to_image(fixed, moving, Linear_en, -1000.0)
```

**Notes:**
- Automatically changes moving image orientation to match fixed
- Output has same dimensions, spacing, origin, and direction as fixed image

---

## HDF5 Storage

### save_med_image

```julia
save_med_image(f::HDF5.File, group_name::String, image::MedImage)
```

Save a MedImage to an HDF5 file.

**Arguments:**
- `f::HDF5.File`: Open HDF5 file handle
- `group_name::String`: Group name within the file
- `image::MedImage`: Image to save

**Returns:**
- `String`: Unique dataset name (UUID) used for storage

**Example:**
```julia
using HDF5

h5open("images.h5", "w") do f
    dataset_name = save_med_image(f, "ct_scans", ct_image)
    println("Saved as: $dataset_name")
end
```

---

### load_med_image

```julia
load_med_image(f::HDF5.File, group_name::String, dataset_name::String)
```

Load a MedImage from an HDF5 file.

**Arguments:**
- `f::HDF5.File`: Open HDF5 file handle
- `group_name::String`: Group name within the file
- `dataset_name::String`: Dataset name (UUID from save_med_image)

**Returns:**
- `MedImage`: Loaded image with all metadata

**Example:**
```julia
using HDF5

h5open("images.h5", "r") do f
    image = load_med_image(f, "ct_scans", "550e8400-e29b-41d4-a716-446655440000")
end
```

---

## Enumerations

### Image_type

```julia
@enum Image_type begin
    MRI_type    # Magnetic Resonance Imaging
    PET_type    # Positron Emission Tomography
    CT_type     # Computed Tomography
end
```

### Image_subtype

```julia
@enum Image_subtype begin
    CT_subtype      # Standard CT
    ADC_subtype     # Apparent Diffusion Coefficient (MRI)
    DWI_subtype     # Diffusion Weighted Imaging (MRI)
    T1_subtype      # T1-weighted MRI
    T2_subtype      # T2-weighted MRI
    FLAIR_subtype   # Fluid-attenuated inversion recovery (MRI)
    FDG_subtype     # Fluorodeoxyglucose PET
    PSMA_subtype    # Prostate-Specific Membrane Antigen PET
end
```

### Interpolator_enum

```julia
@enum Interpolator_enum begin
    Nearest_neighbour_en   # Nearest neighbor - fast, preserves values
    Linear_en              # Linear/trilinear - smooth, good balance
    B_spline_en            # B-spline - smoothest, computationally intensive
end
```

**Usage Guidance:**
- `Nearest_neighbour_en`: Best for label maps, segmentation masks
- `Linear_en`: Best for general image processing
- `B_spline_en`: Best when smoothness is critical

### current_device_enum

```julia
@enum current_device_enum begin
    CPU_current_device       # Standard CPU processing
    CUDA_current_device      # NVIDIA CUDA GPU
    AMD_current_device       # AMD GPU (ROCm)
    ONEAPI_current_device    # Intel oneAPI
end
```

### Mode_mi

```julia
@enum Mode_mi begin
    pixel_array_mode = 0    # Modify pixel array only
    spat_metadata_mode = 2  # Modify spatial metadata only
    all_mode = 3            # Modify both
end
```

### Orientation_code

```julia
@enum Orientation_code begin
    ORIENTATION_RPI    # Right-Posterior-Inferior
    ORIENTATION_LPI    # Left-Posterior-Inferior
    ORIENTATION_RAI    # Right-Anterior-Inferior
    ORIENTATION_LAI    # Left-Anterior-Inferior
    ORIENTATION_RPS    # Right-Posterior-Superior
    ORIENTATION_LPS    # Left-Posterior-Superior
    ORIENTATION_RAS    # Right-Anterior-Superior
    ORIENTATION_LAS    # Left-Anterior-Superior
end
```

**Orientation Conventions:**
- First letter: Left (L) or Right (R) - patient's left/right
- Second letter: Anterior (A) or Posterior (P) - front/back
- Third letter: Superior (S) or Inferior (I) - head/feet

---

## Orientation Utilities

### Orientation Dictionaries

The `Orientation_dicts` module provides mappings between orientation representations:

```julia
# Convert enum to string
orientation_enum_to_string[ORIENTATION_LPS]  # Returns "LPS"

# Convert string to enum
string_to_orientation_enum["RAS"]  # Returns ORIENTATION_RAS

# Get direction cosines for an orientation
orientation_dict_enum_to_number[ORIENTATION_LPS]

# Get orientation from direction cosines
number_to_enum_orientation_dict[direction_tuple]
```

---

## Coordinate System Notes

### Index vs Physical Coordinates

MedImages.jl maintains compatibility with both Julia's array conventions and medical imaging standards:

| Aspect | Julia Arrays | Physical Space |
|--------|--------------|----------------|
| Dimension Order | (dim1, dim2, dim3) | (x, y, z) |
| First Dimension | Fastest varying | X-axis |
| Indexing | 1-based | 0-based (SimpleITK) |

### Coordinate Transformation

```julia
# Transform voxel index to physical point
physical_point = transformIndexToPhysicalPoint_Julia(image, (i, j, k))

# Get image center in voxel coordinates
voxel_center = get_voxel_center_Julia(image.voxel_data)

# Get image center in physical coordinates
physical_center = get_real_center_Julia(image)
```

---

## See Also

- [Getting Started](manual/get_started.md) - Installation and basic usage
- [Tutorials](manual/tutorials.md) - Step-by-step examples
- [Data Structures](reference/data_structures.md) - Detailed type documentation
