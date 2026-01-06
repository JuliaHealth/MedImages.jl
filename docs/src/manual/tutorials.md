# Tutorials

This section provides step-by-step tutorials for common medical image processing workflows using MedImages.jl.

## Tutorial 1: Loading and Inspecting Images

Learn the basics of loading medical images and exploring their properties.

### Loading a NIfTI Image

```julia
using MedImages

# Load a CT scan
ct_image = load_image("/path/to/ct_scan.nii.gz", "CT")

# Inspect basic properties
println("Image dimensions: ", size(ct_image.voxel_data))
println("Voxel spacing (mm): ", ct_image.spacing)
println("Origin (mm): ", ct_image.origin)
println("Direction cosines: ", ct_image.direction)
```

### Understanding the Coordinate System

MedImages.jl stores images using the following conventions:

1. **Voxel data**: Julia array with dimensions (dim1, dim2, dim3), which physically corresponds to (Z, Y, X)
2. **Spatial metadata**: Uses (x, y, z) ordering, consistent with SimpleITK and ITK

```julia
# Get physical dimensions
physical_extent = size(ct_image.voxel_data) .* reverse(ct_image.spacing)
println("Physical extent (x,y,z) in mm: ", physical_extent)

# Access a specific voxel
voxel_value = ct_image.voxel_data[50, 100, 200]  # [z, y, x] in array indices
```

### Saving Images

```julia
# Save as NIfTI
create_nii_from_medimage(ct_image, "/output/processed_ct")
# This creates: /output/processed_ct.nii.gz
```

---

## Tutorial 2: Image Transformations

Learn how to apply spatial transformations to medical images.

### Cropping

Extract a region of interest from an image:

```julia
using MedImages

# Load image
image = load_image("/path/to/image.nii.gz", "CT")

# Define crop parameters (in Julia array order: dim1, dim2, dim3)
# These are 0-based indices for SimpleITK compatibility
start_indices = (10, 20, 30)  # Starting position
crop_size = (100, 150, 200)   # Size of region to extract

# Perform crop
cropped = crop_mi(image, start_indices, crop_size, Linear_en)

# Verify dimensions
println("Original size: ", size(image.voxel_data))
println("Cropped size: ", size(cropped.voxel_data))
println("Original origin: ", image.origin)
println("Cropped origin: ", cropped.origin)
```

### Padding

Add voxels around an image:

```julia
# Pad with air value (-1000 HU for CT)
pad_beginning = (5, 5, 5)   # Add 5 voxels at the start of each dimension
pad_end = (5, 5, 5)         # Add 5 voxels at the end of each dimension
pad_value = -1000.0         # Air in Hounsfield units

padded = pad_mi(image, pad_beginning, pad_end, pad_value, Linear_en)

println("Padded size: ", size(padded.voxel_data))
```

### Rotation

Rotate an image around a specified axis:

```julia
# Rotate 15 degrees around axis 3 (axial plane rotation)
rotated = rotate_mi(image, 3, 15.0, Linear_en)

# Rotate without cropping to original size
rotated_full = rotate_mi(image, 3, 15.0, Linear_en, false)
```

### Translation

Shift the image origin (metadata-only operation):

```julia
# Translate 10 voxels along axis 1 (X direction)
translated = translate_mi(image, 10, 1, Linear_en)

println("Original origin: ", image.origin)
println("Translated origin: ", translated.origin)
```

### Scaling

Resize an image by a scale factor:

```julia
# Downscale by half (uniform scaling)
downscaled = scale_mi(image, 0.5, Linear_en)

# Non-uniform scaling
scaled = scale_mi(image, (1.0, 0.5, 0.5), Linear_en)

println("Original size: ", size(image.voxel_data))
println("Downscaled size: ", size(downscaled.voxel_data))
```

---

## Tutorial 3: Resampling Operations

Learn how to resample images to different spacings and geometries.

### Resampling to Isotropic Spacing

Convert an image to isotropic (equal) voxel spacing:

```julia
using MedImages

image = load_image("/path/to/anisotropic.nii.gz", "CT")

println("Original spacing: ", image.spacing)
println("Original size: ", size(image.voxel_data))

# Resample to 1mm isotropic
isotropic = resample_to_spacing(image, (1.0, 1.0, 1.0), Linear_en)

println("New spacing: ", isotropic.spacing)
println("New size: ", size(isotropic.voxel_data))
```

### Resampling to a Reference Image

Register one image to another's coordinate space:

```julia
# Load fixed (reference) and moving images
ct_image = load_image("/path/to/ct.nii.gz", "CT")
pet_image = load_image("/path/to/pet.nii.gz", "PET")

# Resample PET to CT space
pet_registered = resample_to_image(ct_image, pet_image, Linear_en)

# Verify alignment
println("CT dimensions: ", size(ct_image.voxel_data))
println("PET registered dimensions: ", size(pet_registered.voxel_data))
println("CT spacing: ", ct_image.spacing)
println("PET registered spacing: ", pet_registered.spacing)
```

---

## Tutorial 4: Orientation Management

Learn how to work with image orientations.

### Understanding Orientations

Medical images use 3-letter codes to describe their orientation:
- **First letter**: Patient's Left (L) or Right (R)
- **Second letter**: Anterior (A) or Posterior (P)
- **Third letter**: Superior (S) or Inferior (I)

Common orientations:
- **LPS**: Left-Posterior-Superior (DICOM default)
- **RAS**: Right-Anterior-Superior (neuroimaging default)

### Changing Orientation

```julia
using MedImages

image = load_image("/path/to/image.nii.gz", "CT")

# Convert to RAS orientation (common for neuroimaging)
ras_image = change_orientation(image, ORIENTATION_RAS)

# Convert to LPS orientation (common for CT/PET)
lps_image = change_orientation(image, ORIENTATION_LPS)
```

### Checking Current Orientation

```julia
# Get orientation from direction cosines
current_orientation = number_to_enum_orientation_dict[image.direction]
orientation_string = orientation_enum_to_string[current_orientation]

println("Current orientation: ", orientation_string)
```

---

## Tutorial 5: HDF5 Storage

Learn how to efficiently store and retrieve medical images using HDF5.

### Saving Multiple Images

```julia
using MedImages
using HDF5

# Load some images
ct = load_image("/path/to/ct.nii.gz", "CT")
pet = load_image("/path/to/pet.nii.gz", "PET")

# Save to HDF5
h5open("patient_data.h5", "w") do f
    ct_id = save_med_image(f, "imaging", ct)
    pet_id = save_med_image(f, "imaging", pet)

    println("CT saved as: ", ct_id)
    println("PET saved as: ", pet_id)

    # Store the IDs for later retrieval
    # You might want to save these to a separate metadata group
end
```

### Loading Images from HDF5

```julia
h5open("patient_data.h5", "r") do f
    # Load using the UUID that was returned during save
    ct_loaded = load_med_image(f, "imaging", "your-ct-uuid-here")

    println("Loaded image dimensions: ", size(ct_loaded.voxel_data))
end
```

### Organizing Patient Data

```julia
h5open("study.h5", "w") do f
    # Create groups for different patients
    for patient_id in ["P001", "P002", "P003"]
        group_name = "patients/$patient_id"

        # Load and save patient images
        ct = load_image("/data/$patient_id/ct.nii.gz", "CT")
        dataset_id = save_med_image(f, group_name, ct)
    end
end
```

---

## Tutorial 6: Custom Processing Pipeline

Build a complete image processing pipeline.

### Example: CT Preprocessing Pipeline

```julia
using MedImages

function preprocess_ct(input_path::String, output_path::String)
    # Load image
    println("Loading image...")
    image = load_image(input_path, "CT")

    # Step 1: Resample to isotropic spacing
    println("Resampling to isotropic spacing...")
    image = resample_to_spacing(image, (1.0, 1.0, 1.0), Linear_en)

    # Step 2: Ensure LPS orientation
    println("Converting to LPS orientation...")
    image = change_orientation(image, ORIENTATION_LPS)

    # Step 3: Crop to region of interest (if needed)
    # This would typically be based on automatic detection
    # For example, removing empty slices

    # Step 4: Apply custom processing to voxel data
    println("Applying custom processing...")
    processed_data = clamp.(image.voxel_data, -1000.0, 3000.0)  # Window CT values
    image = update_voxel_data(image, processed_data)

    # Save result
    println("Saving processed image...")
    create_nii_from_medimage(image, output_path)

    println("Done!")
    return image
end

# Usage
processed = preprocess_ct("/input/raw_ct.nii.gz", "/output/processed_ct")
```

### Example: Multi-Modal Registration Pipeline

```julia
using MedImages

function register_pet_to_ct(ct_path::String, pet_path::String, output_path::String)
    # Load images
    ct = load_image(ct_path, "CT")
    pet = load_image(pet_path, "PET")

    println("CT dimensions: ", size(ct.voxel_data))
    println("CT spacing: ", ct.spacing)
    println("PET dimensions: ", size(pet.voxel_data))
    println("PET spacing: ", pet.spacing)

    # Resample PET to CT space
    println("Registering PET to CT space...")
    pet_registered = resample_to_image(ct, pet, Linear_en, 0.0)

    # Verify registration
    println("Registered PET dimensions: ", size(pet_registered.voxel_data))
    println("Registered PET spacing: ", pet_registered.spacing)

    # Save registered PET
    create_nii_from_medimage(pet_registered, output_path)

    return pet_registered
end

# Usage
registered_pet = register_pet_to_ct(
    "/data/ct.nii.gz",
    "/data/pet.nii.gz",
    "/output/pet_registered"
)
```

---

## Tutorial 7: Working with Segmentation Masks

Tips for handling segmentation masks and label maps.

### Loading and Processing Masks

```julia
using MedImages

# Load a segmentation mask (use "CT" type for integer labels)
mask = load_image("/path/to/segmentation.nii.gz", "CT")

# Check unique labels
unique_labels = unique(mask.voxel_data)
println("Labels in mask: ", unique_labels)
```

### Resampling Masks

When resampling segmentation masks, use nearest neighbor interpolation to preserve label values:

```julia
# Original mask
mask = load_image("/path/to/mask.nii.gz", "CT")

# Reference image to match
target = load_image("/path/to/target.nii.gz", "CT")

# Use nearest neighbor for masks
mask_resampled = resample_to_image(target, mask, Nearest_neighbour_en)
```

### Creating a Mask from Thresholding

```julia
# Load CT image
ct = load_image("/path/to/ct.nii.gz", "CT")

# Create bone mask (simple thresholding)
bone_threshold = 300.0  # HU value
bone_mask_data = Float64.(ct.voxel_data .> bone_threshold)

# Create new MedImage with mask data
bone_mask = update_voxel_data(ct, bone_mask_data)

# Save mask
create_nii_from_medimage(bone_mask, "/output/bone_mask")
```

---

## Next Steps

- Review the [API Reference](../api.md) for detailed function documentation
- Explore the [Coordinate Systems](coordinate_systems.md) guide for advanced spatial operations
- Check the [Developer Documentation](../devs/image_registration.md) for implementation details
