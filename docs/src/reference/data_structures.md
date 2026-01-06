# Data Structures Reference

This page provides detailed documentation for all data structures and types in MedImages.jl.

## Primary Data Structure

### MedImage

The `MedImage` struct is the central data structure in MedImages.jl. It encapsulates both voxel data and comprehensive metadata for medical images.

```julia
@with_kw mutable struct MedImage
    voxel_data                                   # Multidimensional array
    origin::Tuple{Float64,Float64,Float64}       # Physical origin (x,y,z)
    spacing::Tuple{Float64,Float64,Float64}      # Voxel spacing (x,y,z)
    direction::NTuple{9,Float64}                 # Direction cosines (3x3 flattened)
    image_type::Image_type                       # Modality type
    image_subtype::Image_subtype                 # Specific protocol
    date_of_saving::DateTime = Dates.today()     # Save timestamp
    acquistion_time::DateTime = Dates.now()      # Acquisition timestamp
    patient_id::String                           # Patient identifier
    current_device::current_device_enum = CPU_current_device
    study_uid::String = string(UUIDs.uuid4())    # DICOM study UID
    patient_uid::String = string(UUIDs.uuid4())  # Patient UID
    series_uid::String = string(UUIDs.uuid4())   # Series UID
    study_description::String = " "              # Study description
    legacy_file_name::String = " "               # Original filename
    display_data::Dict{Any,Any} = Dict()         # Visualization parameters
    clinical_data::Dict{Any,Any} = Dict()        # Clinical metadata
    is_contrast_administered::Bool = false       # Contrast agent flag
    metadata::Dict{Any,Any} = Dict()             # Additional metadata
end
```

---

## Field Documentation

### Voxel Data Fields

#### `voxel_data`

**Type:** `Array{T,N}` where T is typically `Float32`, `Float64`, or `Int16`

The multidimensional array containing the image intensity values.

**Dimension Ordering:**
- For 3D images: `(dim1, dim2, dim3)` corresponds to `(Z, Y, X)` in physical space
- For 4D images: `(dim1, dim2, dim3, dim4)` where dim4 is typically time or diffusion direction

**Example:**
```julia
# Access the voxel array
data = image.voxel_data

# Get dimensions
println(size(data))  # e.g., (512, 512, 256)

# Access specific voxel (z=100, y=200, x=150)
value = data[100, 200, 150]
```

---

### Spatial Metadata Fields

#### `origin`

**Type:** `Tuple{Float64, Float64, Float64}`

The physical coordinates (x, y, z) of the first voxel (index [1,1,1]) in millimeters.

**Example:**
```julia
println(image.origin)  # e.g., (-125.0, -180.0, -50.0)
```

#### `spacing`

**Type:** `Tuple{Float64, Float64, Float64}`

The distance between adjacent voxel centers along each axis (x, y, z) in millimeters.

**Example:**
```julia
println(image.spacing)  # e.g., (0.5, 0.5, 1.0)
```

#### `direction`

**Type:** `NTuple{9, Float64}`

The direction cosine matrix flattened in row-major order. This 3x3 matrix defines how the image axes relate to the physical coordinate system.

**Structure:**
```
[d1, d2, d3, d4, d5, d6, d7, d8, d9]

Represents:
| d1 d2 d3 |   | xx xy xz |
| d4 d5 d6 | = | yx yy yz |
| d7 d8 d9 |   | zx zy zz |
```

**Common Values:**
- Identity (LPS): `(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)`
- RAS: `(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)`

---

### Image Classification Fields

#### `image_type`

**Type:** `Image_type` (enum)

The imaging modality. See [Image_type Enum](#image_type) for values.

#### `image_subtype`

**Type:** `Image_subtype` (enum)

The specific imaging protocol or sequence. See [Image_subtype Enum](#image_subtype) for values.

---

### Temporal Metadata Fields

#### `date_of_saving`

**Type:** `DateTime`

The date and time when the image was saved to disk.

**Default:** `Dates.today()`

#### `acquistion_time`

**Type:** `DateTime`

The date and time when the image was originally acquired.

**Default:** `Dates.now()`

---

### Patient Identification Fields

#### `patient_id`

**Type:** `String`

The patient identifier from the source data or assigned during loading.

#### `patient_uid`

**Type:** `String`

A unique identifier for the patient. Automatically generated as UUID if not provided.

**Default:** `string(UUIDs.uuid4())`

---

### Study Identification Fields

#### `study_uid`

**Type:** `String`

The DICOM Study Instance UID or a generated unique identifier.

**Default:** `string(UUIDs.uuid4())`

#### `series_uid`

**Type:** `String`

The DICOM Series Instance UID or a generated unique identifier.

**Default:** `string(UUIDs.uuid4())`

#### `study_description`

**Type:** `String`

A description of the imaging study.

**Default:** `" "`

---

### Processing Fields

#### `current_device`

**Type:** `current_device_enum`

The compute device used for processing this image.

**Default:** `CPU_current_device`

---

### Source File Fields

#### `legacy_file_name`

**Type:** `String`

The original filename from which the image was loaded.

**Default:** `" "`

---

### Extensible Metadata Fields

#### `display_data`

**Type:** `Dict{Any, Any}`

A dictionary for storing visualization-related parameters such as color maps, window/level settings, etc.

**Example:**
```julia
image.display_data["window_center"] = 40
image.display_data["window_width"] = 400
image.display_data["colormap"] = "gray"
```

#### `clinical_data`

**Type:** `Dict{Any, Any}`

A dictionary for storing patient clinical information.

**Example:**
```julia
image.clinical_data["age"] = 65
image.clinical_data["sex"] = "M"
image.clinical_data["weight_kg"] = 75.0
image.clinical_data["diagnosis"] = "Normal"
```

#### `metadata`

**Type:** `Dict{Any, Any}`

A dictionary for storing any additional metadata not covered by other fields.

**Example:**
```julia
image.metadata["source_format"] = "DICOM"
image.metadata["reconstruction_kernel"] = "B30f"
image.metadata["slice_thickness"] = 1.0
```

#### `is_contrast_administered`

**Type:** `Bool`

Flag indicating whether a contrast agent was administered during imaging.

**Default:** `false`

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

**Usage:**
```julia
image.image_type = CT_type

if image.image_type == CT_type
    println("Processing CT scan")
end
```

---

### Image_subtype

```julia
@enum Image_subtype begin
    CT_subtype      # Standard CT scan
    ADC_subtype     # Apparent Diffusion Coefficient (MRI)
    DWI_subtype     # Diffusion Weighted Imaging (MRI)
    T1_subtype      # T1-weighted MRI
    T2_subtype      # T2-weighted MRI
    FLAIR_subtype   # Fluid-attenuated inversion recovery (MRI)
    FDG_subtype     # Fluorodeoxyglucose PET
    PSMA_subtype    # Prostate-Specific Membrane Antigen PET
end
```

**Typical Associations:**
| Image Type | Common Subtypes |
|------------|-----------------|
| `CT_type` | `CT_subtype` |
| `MRI_type` | `T1_subtype`, `T2_subtype`, `FLAIR_subtype`, `ADC_subtype`, `DWI_subtype` |
| `PET_type` | `FDG_subtype`, `PSMA_subtype` |

---

### Interpolator_enum

```julia
@enum Interpolator_enum begin
    Nearest_neighbour_en   # Nearest neighbor interpolation
    Linear_en              # Linear/trilinear interpolation
    B_spline_en            # B-spline interpolation
end
```

**Selection Guide:**

| Interpolator | Speed | Quality | Best For |
|--------------|-------|---------|----------|
| `Nearest_neighbour_en` | Fastest | Preserves discrete values | Segmentation masks, label maps |
| `Linear_en` | Fast | Good, may blur edges | General intensity images |
| `B_spline_en` | Slowest | Smoothest | High-quality resampling |

---

### current_device_enum

```julia
@enum current_device_enum begin
    CPU_current_device       # Standard CPU processing
    CUDA_current_device      # NVIDIA CUDA GPU
    AMD_current_device       # AMD GPU (ROCm)
    ONEAPI_current_device    # Intel oneAPI device
end
```

**Note:** GPU support requires appropriate packages (CUDA.jl, AMDGPU.jl, oneAPI.jl) to be installed and configured.

---

### Mode_mi

```julia
@enum Mode_mi begin
    pixel_array_mode = 0    # Modify pixel array only
    spat_metadata_mode = 2  # Modify spatial metadata only
    all_mode = 3            # Modify both
end
```

This enum is used internally to control which aspects of an image are modified during transformations.

---

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

**Letter Meanings:**
- **First letter:** L (Left) or R (Right) - direction of increasing X
- **Second letter:** A (Anterior) or P (Posterior) - direction of increasing Y
- **Third letter:** S (Superior) or I (Inferior) - direction of increasing Z

**Common Conventions:**
- **LPS:** DICOM standard, used by ITK, SimpleITK
- **RAS:** Neuroimaging standard, used by FreeSurfer, FSL

---

### CoordinateTerms

```julia
@enum CoordinateTerms begin
    ITK_COORDINATE_UNKNOWN = 0
    ITK_COORDINATE_Right = 2
    ITK_COORDINATE_Left = 3
    ITK_COORDINATE_Posterior = 4
    ITK_COORDINATE_Anterior = 5
    ITK_COORDINATE_Inferior = 8
    ITK_COORDINATE_Superior = 9
end
```

Used internally for ITK compatibility.

---

### CoordinateMajornessTerms

```julia
@enum CoordinateMajornessTerms begin
    PrimaryMinor = 0      # Primary minor coordinate
    SecondaryMinor = 8    # Secondary minor coordinate
    TertiaryMinor = 16    # Tertiary minor coordinate
end
```

Used internally for coordinate system transformations.

---

## Constructor and Factory Functions

### Default Constructor

```julia
# Using keyword arguments (recommended)
image = MedImage(
    voxel_data = zeros(Float64, 256, 256, 100),
    origin = (0.0, 0.0, 0.0),
    spacing = (1.0, 1.0, 1.0),
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    image_type = CT_type,
    image_subtype = CT_subtype,
    patient_id = "PATIENT001"
)
```

### Using load_image

```julia
# Load from file (recommended approach)
image = load_image("/path/to/image.nii.gz", "CT")
```

### Using update Functions

```julia
# Create modified copy with new voxel data
new_image = update_voxel_data(original_image, new_data)

# Create modified copy with new voxel and spatial data
new_image = update_voxel_and_spatial_data(
    original_image,
    new_data,
    new_origin,
    new_spacing,
    new_direction
)
```

---

## Memory Layout

The `MedImage` struct uses Julia's default memory layout:

- **Voxel data:** Stored as a contiguous Julia array (column-major order)
- **Tuples:** Stored inline in the struct
- **Strings:** Stored as references to heap-allocated strings
- **Dicts:** Stored as references to heap-allocated dictionaries

**Memory Considerations:**

```julia
# Approximate memory usage
voxel_memory = sizeof(eltype(image.voxel_data)) * prod(size(image.voxel_data))
overhead = 8 * 3 * 3  # origin, spacing (tuples)
           + 8 * 9    # direction (tuple)
           + 8 * 2    # DateTime fields
           + 8 * 10   # String references
           + 8 * 3    # Dict references
           + 8 * 3    # Enum fields

total_approximate = voxel_memory + overhead
```

---

## Serialization

### To HDF5

```julia
using HDF5

h5open("output.h5", "w") do f
    uuid = save_med_image(f, "group_name", image)
end
```

### To NIfTI

```julia
create_nii_from_medimage(image, "/path/to/output")
```

### To JSON (metadata only)

```julia
using JSON

metadata = Dict(
    "origin" => image.origin,
    "spacing" => image.spacing,
    "direction" => image.direction,
    "image_type" => string(image.image_type),
    "dimensions" => size(image.voxel_data)
)

json_string = JSON.json(metadata)
```
