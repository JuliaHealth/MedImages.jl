
module Load_and_save
using Dictionaries, Dates, PyCall
using Accessors, UUIDs, ITKIOWrapper
using ..MedImage_data_struct
using ..MedImage_data_struct: MedImage, BatchedMedImage
using ..Brute_force_orientation
using ..Utils
export load_image
export update_voxel_and_spatial_data
export update_voxel_data

"""
helper function for dicom #2
returns an array of pixel data for unique ids within dicom files
"""
function get_pixel_data(dicom_data_array)
  if length(dicom_data_array) == 1
    #in case e have 2D image
    return only(dicom_data_array).PixelData
  else
    #in case we have 3D image
    return cat([dcm.PixelData for dcm in dicom_data_array]...; dims=3)
  end
end


"""
Note: No long supported
Determines the study type for an image based on its metadata using SimpleITK
"""
# function infer_modality(image)
#   sitk = pyimport("SimpleITK")
#   metadata = Dict()
#   for key in image.GetMetaDataKeys()
#     metadata[key] = image.GetMetaData(key)
#   end

#   # Check for CT-specific metadata
#   if "modality" in keys(metadata) && metadata["modality"] == "CT"
#     return "CT"
#   end

#   spacing = image.GetSpacing()
#   size = image.GetSize()
#   pixel_type = image.GetPixelIDTypeAsString()

#   # Relaxed spacing condition for CT
#   spacing_condition_ct = all(s -> s <= 1.0, spacing)
#   pixel_type_condition_ct = pixel_type == "16-bit signed integer"

#   if spacing_condition_ct && pixel_type_condition_ct
#     # Additional check: sample some voxels to see if they're in the typical CT range
#     stats = sitk.StatisticsImageFilter()
#     stats.Execute(image)
#     mean_value = stats.GetMean()

#     if -1000 <= mean_value <= 3000  # Typical range for CT in Hounsfield units
#       return MedImage_data_struct.CT_type
#     end
#   end

#   # PET condition (unchanged)
#   spacing_condition_pet = all(s -> s > 1.0, spacing)
#   pixel_type_condition_pet = occursin("float", lowercase(pixel_type))

#   if spacing_condition_pet && pixel_type_condition_pet
#     return MedImage_data_struct.PET_type
#   end

#   return MedImage_data_struct.CT_type
# end

"""
    create_nii_from_medimage(med_image, file_path, extension=".nii.gz")

Save a MedImage object to a NIfTI file.

This function exports a MedImage struct to the standard NIfTI file format,
preserving all spatial metadata (origin, spacing, direction) in the NIfTI header.

# Arguments
- `med_image::MedImage`: MedImage object to save
- `file_path::String`: Destination file path
- `extension::String=".nii.gz"`: File extension (default: .nii.gz for compressed NIfTI)

# Returns
- Nothing: Saves file to disk

# Examples
```julia
# Create a MedImage
img = MedImage(rand(10, 10, 10), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

# Save as NIfTI file
create_nii_from_medimage(img, "output_image")

# Saves as "output_image.nii.gz" (compressed)
# Can be loaded by other NIfTI readers (ITK-SNAP, Slicer, etc.)

# Save with different extension
create_nii_from_medimage(img, "output_image", ".nii")
# Saves as "output_image.nii" (uncompressed)
```

# Notes
- Uses ITKIOWrapper.jl for NIfTI I/O operations
- Preserves all spatial metadata in NIfTI header fields
- Default compression (.nii.gz) reduces file size significantly
- Resulting files are compatible with standard medical imaging software
- Useful for sharing results or intermediate processing steps
"""
function create_nii_from_medimage(med_image::MedImage, file_path::String, extension::String=".nii.gz")
  # Ensure path has extension if not already present
  full_path = endswith(file_path, extension) ? file_path : file_path * extension

  # Prepare VoxelData (requires Float32 for ITKIOWrapper)
  voxel_f32 = Array{Float32, 3}(med_image.voxel_data)
  vd = ITKIOWrapper.DataStructs.VoxelData(voxel_f32)

  # Prepare SpatialMetaData
  origin = NTuple{3, Float64}(med_image.origin)
  spacing = NTuple{3, Float64}(med_image.spacing)
  sz = NTuple{3, Int64}(size(med_image.voxel_data))
  direction = NTuple{9, Float64}(med_image.direction)
  
  meta = ITKIOWrapper.DataStructs.SpatialMetaData(origin, spacing, sz, direction)

  # Save using native ITK wrapper
  ITKIOWrapper.save_image(vd, meta, full_path, false)
end

"""
    update_voxel_data(old_image::MedImage, new_voxel_data::AbstractArray)

Update the voxel data of a `MedImage` while preserving all other metadata.

This function creates a new `MedImage` with updated `voxel_data` using the `@set` macro
from `Accessors.jl`. It is the preferred way to "modify" the immutable `MedImage` struct.

# Arguments
- `old_image::MedImage`: Original image object.
- `new_voxel_data::AbstractArray`: New array for the `voxel_data` field.

# Returns
- `MedImage`: A new image object with updated data.
"""
function update_voxel_data(old_image::MedImage, new_voxel_data::AbstractArray)
  return @set old_image.voxel_data = new_voxel_data
end

function update_voxel_data(old_image::BatchedMedImage, new_voxel_data::AbstractArray)
  return @set old_image.voxel_data = new_voxel_data
end

"""
    update_voxel_and_spatial_data(old_image, new_voxel_data::AbstractArray, new_origin, new_spacing, new_direction=nothing)

Update both voxel data and spatial metadata (origin, spacing, direction) of a `MedImage`.

This function uses the `@set` macro to create a new `MedImage` with multiple fields
updated. If `new_direction` is not provided, the original direction is kept.

# Arguments
- `old_image`: The image object to update.
- `new_voxel_data`: Target voxel data array.
- `new_origin`: Target origin tuple/vector.
- `new_spacing`: Target spacing tuple/vector.
- `new_direction`: (Optional) Target direction matrix.

# Returns
- `MedImage`: A new image object with updated data and spatial metadata.
"""
function update_voxel_and_spatial_data(old_image, new_voxel_data::AbstractArray, new_origin, new_spacing, new_direction=nothing)
    res = @set old_image.voxel_data = new_voxel_data
    res = @set res.origin = Utils.ensure_tuple(new_origin)
    res = @set res.spacing = Utils.ensure_tuple(new_spacing)
    if !isnothing(new_direction)
        res = @set res.direction = Utils.ensure_tuple(new_direction)
    end
    return res
end


"""
    load_image(path, type)

Load a medical image from file (DICOM or NIfTI format) into a MedImage struct.

This function supports both DICOM and NIfTI formats, converting them into the unified
MedImage struct representation. Both formats load into the same structure, making
downstream processing format-agnostic.

# Arguments
- `path::String`: Path to the image file or DICOM directory
- `type::String`: Image type identifier ("dicom" or "nifti")

# Returns
- `MedImage`: Loaded medical image in standard MedImage format

# Examples
```julia
# Load a NIfTI file
julia> nifti_img = load_image("brain.nii.gz", "nifti")
MedImage(...)

# Load a DICOM series from directory
julia> dicom_img = load_image("/path/to/dicom/series/", "dicom")
MedImage(...)

# Both return the same MedImage struct format
julia> typeof(nifti_img) == typeof(dicom_img)
true
```

# Notes
- **Format unification**: Both DICOM and NIfTI load into identical MedImage structs
- **Orientation**: By default, ITKIOWrapper loads images in LPS orientation
- **Modality inference**: The automatic modality inference has been sunsetted;
  users must explicitly specify image type downstream
- **Standardization**: Setting all files to a standard orientation (e.g., RAS) at
  the beginning is recommended for consistency across datasets
- **Metadata**: DICOM metadata is extracted and stored in the `metadata` field
"""
function load_image(path::String, type::String)::MedImage
  # ITKIOWrapper I/O defaults to LPS orientation for any given image
  # The section for inferring modality has now been sunsetted, with user having to explicitly pass the modality downstream
  imaging_study = ITKIOWrapper.load_image(path)
  spatial_meta = ITKIOWrapper.load_spatial_metadata(imaging_study)
  voxel_arr_struct = ITKIOWrapper.load_voxel_data(imaging_study, spatial_meta)
  voxel_arr = voxel_arr_struct.dat
  #voxel_arr = permutedims(voxel_arr, (3, 2, 1))
  study_type = type == "CT" ? MedImage_data_struct.CT_type : MedImage_data_struct.PET_type
  subtype = type == "CT" ? MedImage_data_struct.CT_subtype : MedImage_data_struct.FDG_subtype
  legacy_file_name_field = string(split(path, "/")[length(split(path, "/"))])

  metadata_dict = Dict{Any, Any}()
  try
      metadata_dict = _get_metadata(path)
  catch e
      @warn "Could not extract DICOM metadata using pydicom: $e"
  end

  return MedImage(voxel_data=voxel_arr, origin=spatial_meta.origin, spacing=spatial_meta.spacing, direction=spatial_meta.direction, patient_id="test_id", image_type=study_type, image_subtype=subtype, legacy_file_name=legacy_file_name_field, metadata=metadata_dict)
end

function _get_metadata(path::String)
    pydicom = pyimport("pydicom")
    os = pyimport("os")

    target_path = path
    if isdir(path)
        # Find first DICOM file in directory
        files = readdir(path)
        # Try to find .dcm extension
        dcm_files = filter(f -> endswith(lowercase(f), ".dcm"), files)
        if !isempty(dcm_files)
            target_path = joinpath(path, dcm_files[1])
        else
            # If no .dcm, try first file that is not a directory
            for f in files
                full_p = joinpath(path, f)
                if isfile(full_p)
                    target_path = full_p
                    break
                end
            end
        end
    end

    if !isfile(target_path)
        return Dict{Any, Any}()
    end

    try
        ds = pydicom.dcmread(target_path, stop_before_pixels=true)
        return _pydicom_ds_to_dict(ds)
    catch e
        # If not a DICOM file or other error
        return Dict{Any, Any}()
    end
end

function _pydicom_ds_to_dict(ds)
    d = Dict{Any, Any}()
    for elem in ds
        key = elem.keyword
        if isempty(key)
            # Use tag if keyword is empty (e.g. private tags)
            key = string(elem.tag)
        end

        val = elem.value

        # Handle Sequences
        if elem.VR == "SQ"
            seq_list = []
            for item in val
                push!(seq_list, _pydicom_ds_to_dict(item))
            end
            d[key] = seq_list
        # Handle MultiValue
        elseif typeof(val) <: PyObject && pybuiltin("isinstance")(val, pyimport("pydicom.multival.MultiValue"))
             d[key] = collect(val)
        else
             d[key] = val
        end
    end
    return d
end

end



