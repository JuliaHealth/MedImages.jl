# Image Registration Developer Guide

This guide provides technical details for developers working on image registration features in MedImages.jl.

## Architecture Overview

MedImages.jl implements image registration through the `Resample_to_target` module. The registration workflow involves:

1. Orientation alignment
2. Coordinate transformation
3. Interpolation at target grid points

```
Moving Image --> Orientation Change --> Coordinate Mapping --> Interpolation --> Registered Image
                       |                      |                      |
                       v                      v                      v
                  Match fixed           Map moving to          Sample at fixed
                  orientation           fixed space             grid points
```

---

## Core Registration Function

### resample_to_image

**Location:** `src/Resample_to_target.jl`

The main registration function that resamples a moving image to match a fixed image's geometry.

```julia
function resample_to_image(im_fixed::MedImage, im_moving::MedImage,
                           interpolator_enum::Interpolator_enum,
                           value_to_extrapolate=Nothing)::MedImage
```

**Implementation Steps:**

1. **Extrapolation Value Determination:**
   If not provided, uses median of fixed image corner values.

2. **Orientation Alignment:**
   Changes moving image orientation to match fixed image.

3. **Grid Generation:**
   Creates target grid based on fixed image dimensions.

4. **Coordinate Transformation:**
   Maps target grid points to moving image coordinates.

5. **Interpolation:**
   Samples moving image at transformed coordinates.

---

## Coordinate Transformation Details

### Grid Point Calculation

The target grid is generated from the fixed image dimensions:

```julia
new_size = size(im_fixed.voxel_data)
points_to_interpolate = get_base_indicies_arr(new_size)
```

### Origin Difference Handling

The origin difference between images is incorporated into the transformation:

```julia
origin_diff = collect(im_fixed.origin) - collect(im_moving.origin)
points_to_interpolate = points_to_interpolate .+ origin_diff
```

### Spacing Transformation

Points are scaled by the target spacing:

```julia
points_to_interpolate = points_to_interpolate .- 1
points_to_interpolate = points_to_interpolate .* new_spacing
points_to_interpolate = points_to_interpolate .+ 1
```

---

## Interpolation Implementation

### Utils Module Functions

**Location:** `src/Utils.jl`

#### interpolate_my

Batch interpolation function supporting multiple points:

```julia
function interpolate_my(points_to_interpolate, input_array, input_array_spacing,
                        interpolator_enum, keep_begining_same,
                        extrapolate_value=0, use_fast=true)
```

**Parameters:**
- `points_to_interpolate`: Array of 3D coordinates
- `input_array`: Source voxel data
- `input_array_spacing`: Source image spacing
- `interpolator_enum`: Interpolation method
- `keep_begining_same`: Whether to preserve starting indices
- `extrapolate_value`: Value for out-of-bounds points
- `use_fast`: Use optimized interpolation path

#### interpolate_point

Single point interpolation:

```julia
function interpolate_point(point, itp, keep_begining_same=false, extrapolate_value=0)
```

---

## Interpolation Methods

### Nearest Neighbor

Uses `Interpolations.Constant()` for integer-preserving interpolation.

**Characteristics:**
- No new values introduced
- Sharp edges preserved
- Block-like artifacts at large scale factors
- Best for: Segmentation masks, label maps

### Linear/Trilinear

Uses `Interpolations.Linear()` for smooth interpolation.

**Characteristics:**
- Weighted average of surrounding voxels
- Smooth transitions
- May blur sharp features
- Best for: General intensity images

### B-Spline

Uses `Interpolations.BSpline()` for high-quality interpolation.

**Characteristics:**
- Smooth derivative continuity
- Best quality for smooth images
- Computationally more expensive
- Best for: High-quality resampling

---

## GPU Acceleration

### Kernel-Based Resampling

**Location:** `src/Utils.jl`

GPU acceleration is implemented using KernelAbstractions.jl:

```julia
function resample_kernel_launch(image_data, old_spacing, new_spacing,
                                 new_dims, interpolator_enum)
```

**Supported Backends:**
- CPU (default)
- CUDA (NVIDIA GPUs)
- AMD (ROCm)
- oneAPI (Intel)

### Kernel Implementation

Two primary kernels:

1. **Trilinear Kernel:** For `Linear_en` interpolation
2. **Nearest Neighbor Kernel:** For `Nearest_neighbour_en` interpolation

---

## Orientation Change Implementation

### Spatial_metadata_change Module

**Location:** `src/Spatial_metadata_change.jl`

#### change_orientation

```julia
function change_orientation(im::MedImage, new_orientation::Orientation_code)::MedImage
```

**Implementation:**

1. **Lookup Transformation:**
   Uses pre-computed transformation from `orientation_pair_to_operation_dict`

2. **Apply Permutation:**
   Reorders array dimensions using `permutedims`

3. **Apply Reversals:**
   Flips axes as needed using `reverse`

4. **Update Origin:**
   Recalculates origin based on transformation

5. **Update Spacing:**
   Reorders spacing to match new orientation

---

## Pre-computed Orientation Mappings

### Orientation_dicts Module

**Location:** `src/Orientation_dicts.jl`

Contains dictionaries mapping between:

1. **Orientation codes and strings:**
   ```julia
   orientation_enum_to_string[ORIENTATION_LPS]  # "LPS"
   string_to_orientation_enum["RAS"]  # ORIENTATION_RAS
   ```

2. **Orientations and direction cosines:**
   ```julia
   orientation_dict_enum_to_number[ORIENTATION_LPS]  # direction tuple
   number_to_enum_orientation_dict[direction_tuple]  # Orientation_code
   ```

3. **Orientation pairs and transformations:**
   ```julia
   orientation_pair_to_operation_dict[(old_orient, new_orient)]
   # Returns: (permutation, reverse_axes, origin_transforms, spacing_transforms)
   ```

---

## Testing Registration

### Test Files

Registration tests are located in `test/resample_to_target_tests/`

### Test Approach

1. Load fixed and moving images
2. Apply registration using both MedImages.jl and SimpleITK
3. Compare spatial metadata (origin, spacing, direction)
4. Optionally compare voxel values (with tolerance)

### Key Test Considerations

- Different interpolation implementations produce different values
- Metadata comparison is more reliable than voxel comparison
- Use `skip_voxel_comparison=true` for algorithm-dependent operations

---

## Performance Considerations

### Memory Usage

Registration creates new arrays rather than modifying in-place:

```julia
# Each step creates a new MedImage
aligned = change_orientation(moving, fixed_orientation)
resampled = resample_to_image(fixed, aligned, interp)
```

**Optimization Tip:** For very large images, consider processing in chunks.

### Computational Complexity

- **Grid generation:** O(n) where n = total voxels
- **Interpolation:** O(n) with constant factor per voxel
- **Orientation change:** O(n) for permutation and reversal

### Parallel Processing

The interpolation step is embarrassingly parallel:

```julia
# Points can be processed independently
for point in points_to_interpolate
    result[point] = interpolate(moving, point)
end
```

---

## Extending Registration

### Adding New Interpolation Methods

1. Add enum value to `Interpolator_enum` in `MedImage_data_struct.jl`
2. Add case handling in `interpolate_my` in `Utils.jl`
3. Optionally add GPU kernel in `Utils.jl`
4. Add tests in `test/` directory

### Adding Transform Types

Currently supports identity transform only. To add more:

1. Create transform struct (e.g., `AffineTransform`)
2. Implement `apply_transform(transform, point)` function
3. Integrate into `resample_to_image` workflow

---

## Related Modules

- `Basic_transformations`: Rotation, translation, scaling
- `Spatial_metadata_change`: Orientation and resampling operations
- `Utils`: Core interpolation functions
- `Load_and_save`: Image I/O operations

---

## References

- ITK Software Guide: [itk.org](https://itk.org/ItkSoftwareGuide.pdf)
- SimpleITK Documentation: [simpleitk.readthedocs.io](https://simpleitk.readthedocs.io)
- Julia Interpolations.jl: [github.com/JuliaMath/Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl)
