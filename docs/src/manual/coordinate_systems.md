# Coordinate Systems Guide

This guide explains the coordinate system conventions used in MedImages.jl and how they relate to medical imaging standards.

## Overview

Medical imaging involves multiple coordinate systems:

1. **Array Index Space**: How voxels are stored in memory
2. **Physical Space**: Real-world coordinates in millimeters
3. **Patient Space**: Anatomical directions (Left/Right, Anterior/Posterior, Superior/Inferior)

Understanding these relationships is crucial for correct spatial operations.

---

## Array Index Space

### Julia Array Conventions

Julia uses column-major ordering, which affects how 3D medical images are stored:

```
image.voxel_data[dim1, dim2, dim3]
                  |      |      |
                  Z      Y      X  (physical mapping)
```

**Key Points:**
- First dimension (dim1) corresponds to Z (axial slices)
- Second dimension (dim2) corresponds to Y (coronal direction)
- Third dimension (dim3) corresponds to X (sagittal direction)
- Indices are 1-based in Julia

**Example:**
```julia
# Access voxel at physical position (x=50, y=100, z=25)
# In Julia array indexing:
value = image.voxel_data[25, 100, 50]  # [z, y, x]
```

---

## Physical Space

### Origin and Spacing

Physical coordinates are defined by the `origin` and `spacing` fields:

```julia
# Origin: Physical coordinates of voxel [1,1,1] in mm
image.origin = (x0, y0, z0)

# Spacing: Distance between voxels in mm
image.spacing = (dx, dy, dz)
```

**Note:** Origin and spacing use (x, y, z) ordering, consistent with ITK and SimpleITK.

### Index to Physical Point Transformation

The transformation from array index to physical coordinates:

```
Physical_X = Origin_X + (index_X - 1) * Spacing_X * Direction[1:3]
Physical_Y = Origin_Y + (index_Y - 1) * Spacing_Y * Direction[4:6]
Physical_Z = Origin_Z + (index_Z - 1) * Spacing_Z * Direction[7:9]
```

**Using MedImages.jl:**
```julia
# Transform voxel index to physical point
physical_point = transformIndexToPhysicalPoint_Julia(image, (i, j, k))
```

---

## Direction Cosine Matrix

The `direction` field is a 9-element tuple representing a 3x3 rotation matrix:

```
direction = (d1, d2, d3, d4, d5, d6, d7, d8, d9)

Matrix form:
| d1 d2 d3 |   Defines how image axes
| d4 d5 d6 |   align with physical axes
| d7 d8 d9 |
```

### Common Direction Matrices

**Identity (LPS orientation):**
```julia
direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
```
- Image X-axis aligned with physical X (Left)
- Image Y-axis aligned with physical Y (Posterior)
- Image Z-axis aligned with physical Z (Superior)

**RAS orientation:**
```julia
direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
```
- Image X-axis points Right (opposite of physical X)
- Image Y-axis points Anterior (opposite of physical Y)
- Image Z-axis points Superior (same as physical Z)

---

## Patient Coordinate System

### Anatomical Directions

Medical imaging uses patient-relative directions:

| Direction | Abbreviation | Description |
|-----------|--------------|-------------|
| Left | L | Patient's left side |
| Right | R | Patient's right side |
| Anterior | A | Front of patient |
| Posterior | P | Back of patient |
| Superior | S | Towards head |
| Inferior | I | Towards feet |

### Orientation Codes

MedImages.jl uses 3-letter orientation codes:

```julia
@enum Orientation_code begin
    ORIENTATION_RPI  # X+ = Right,    Y+ = Posterior, Z+ = Inferior
    ORIENTATION_LPI  # X+ = Left,     Y+ = Posterior, Z+ = Inferior
    ORIENTATION_RAI  # X+ = Right,    Y+ = Anterior,  Z+ = Inferior
    ORIENTATION_LAI  # X+ = Left,     Y+ = Anterior,  Z+ = Inferior
    ORIENTATION_RPS  # X+ = Right,    Y+ = Posterior, Z+ = Superior
    ORIENTATION_LPS  # X+ = Left,     Y+ = Posterior, Z+ = Superior
    ORIENTATION_RAS  # X+ = Right,    Y+ = Anterior,  Z+ = Superior
    ORIENTATION_LAS  # X+ = Left,     Y+ = Anterior,  Z+ = Superior
end
```

**Reading Orientation Codes:**
- First letter: Direction of increasing X
- Second letter: Direction of increasing Y
- Third letter: Direction of increasing Z

---

## Common Conventions

### DICOM / ITK / SimpleITK (LPS)

The default standard for most medical imaging:

- **X+** points to patient's Left
- **Y+** points to Posterior
- **Z+** points to Superior

```julia
# MedImages.jl default (via ITKIOWrapper)
image = load_image("scan.nii.gz", "CT")  # Loaded in LPS
```

### Neuroimaging (RAS)

Common in brain imaging software (FreeSurfer, FSL, MNI):

- **X+** points to patient's Right
- **Y+** points to Anterior
- **Z+** points to Superior

```julia
# Convert to RAS for neuroimaging
ras_image = change_orientation(image, ORIENTATION_RAS)
```

---

## Coordinate Transformations

### Changing Orientation

```julia
using MedImages

# Load image (default LPS)
image = load_image("scan.nii.gz", "CT")

# Check current orientation
current_orient = number_to_enum_orientation_dict[image.direction]
println("Current: ", orientation_enum_to_string[current_orient])

# Change to RAS
ras_image = change_orientation(image, ORIENTATION_RAS)
```

**What happens during orientation change:**
1. Voxel array is permuted to reorder dimensions
2. Axes may be reversed (flipped)
3. Origin is recalculated
4. Spacing is reordered
5. Direction matrix is updated

### Manual Coordinate Conversion

```julia
# Voxel index to physical coordinates
function voxel_to_physical(image, voxel_idx)
    # voxel_idx is (z, y, x) in Julia order
    x = image.origin[1] + (voxel_idx[3] - 1) * image.spacing[1]
    y = image.origin[2] + (voxel_idx[2] - 1) * image.spacing[2]
    z = image.origin[3] + (voxel_idx[1] - 1) * image.spacing[3]
    return (x, y, z)
end

# Physical coordinates to voxel index
function physical_to_voxel(image, physical_pt)
    x_idx = round(Int, (physical_pt[1] - image.origin[1]) / image.spacing[1]) + 1
    y_idx = round(Int, (physical_pt[2] - image.origin[2]) / image.spacing[2]) + 1
    z_idx = round(Int, (physical_pt[3] - image.origin[3]) / image.spacing[3]) + 1
    return (z_idx, y_idx, x_idx)  # Julia array order
end
```

---

## Practical Examples

### Example 1: Finding Image Center

```julia
# Voxel center
voxel_center = get_voxel_center_Julia(image.voxel_data)

# Physical center
physical_center = get_real_center_Julia(image)

println("Voxel center: ", voxel_center)
println("Physical center (mm): ", physical_center)
```

### Example 2: Checking Physical Extent

```julia
# Calculate physical extent of image
dims = size(image.voxel_data)
spacing = image.spacing
origin = image.origin

# Physical extent in each direction
extent_x = (dims[3] - 1) * spacing[1]  # dim3 = X
extent_y = (dims[2] - 1) * spacing[2]  # dim2 = Y
extent_z = (dims[1] - 1) * spacing[3]  # dim1 = Z

println("Physical extent (mm):")
println("  X: $(origin[1]) to $(origin[1] + extent_x)")
println("  Y: $(origin[2]) to $(origin[2] + extent_y)")
println("  Z: $(origin[3]) to $(origin[3] + extent_z)")
```

### Example 3: Resampling with Coordinate Preservation

```julia
# Original physical extent
original_extent = size(image.voxel_data) .* reverse(image.spacing)

# Resample to isotropic
resampled = resample_to_spacing(image, (1.0, 1.0, 1.0), Linear_en)

# New extent (should be similar)
new_extent = size(resampled.voxel_data) .* reverse(resampled.spacing)

println("Original extent: ", original_extent)
println("Resampled extent: ", new_extent)
```

---

## Troubleshooting

### Issue: Image Appears Flipped

**Cause:** Incorrect orientation handling between software packages.

**Solution:**
```julia
# Check and correct orientation
current = number_to_enum_orientation_dict[image.direction]
println("Current orientation: ", current)

# Convert to expected orientation
if current != ORIENTATION_LPS
    image = change_orientation(image, ORIENTATION_LPS)
end
```

### Issue: Registration Misalignment

**Cause:** Images have different orientations.

**Solution:**
```julia
# Ensure both images have same orientation before registration
fixed = change_orientation(fixed, ORIENTATION_LPS)
moving = change_orientation(moving, ORIENTATION_LPS)

# Then resample
registered = resample_to_image(fixed, moving, Linear_en)
```

### Issue: Coordinates Don't Match External Software

**Cause:** Different coordinate system conventions.

**Solution:** Check the orientation expected by the external software and convert:

```julia
# For neuroimaging software expecting RAS
ras_image = change_orientation(image, ORIENTATION_RAS)
create_nii_from_medimage(ras_image, "output_for_fsl")
```

---

## Summary Table

| Aspect | Julia Arrays | Physical Space | Notes |
|--------|--------------|----------------|-------|
| Dimension Order | (dim1, dim2, dim3) | (X, Y, Z) | dim1=Z, dim2=Y, dim3=X |
| Index Base | 1-based | N/A | SimpleITK uses 0-based |
| Origin/Spacing | N/A | (x, y, z) tuple | In millimeters |
| Direction | N/A | 9-element tuple | Row-major 3x3 matrix |
| Default Orientation | N/A | LPS | Via ITKIOWrapper |
