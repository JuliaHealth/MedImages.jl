module Load_and_save
using Dictionaries, Dates, PyCall
using Accessors, UUIDs
using ..MedImage_data_struct
using ..MedImage_data_struct: MedImage, BatchedMedImage
using ..Utils
export load_image
export update_voxel_and_spatial_data
export update_voxel_data
export create_nii_from_medimage

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
    create_nii_from_medimage(med_image, file_path, extension=".nii.gz")

Save a MedImage object to a NIfTI file using SimpleITK via PyCall.
"""
function create_nii_from_medimage(med_image::MedImage, file_path::String, extension::String=".nii.gz")
  sitk = pyimport("SimpleITK")
  np = pyimport("numpy")
  
  full_path = endswith(file_path, extension) ? file_path : file_path * extension

  # Permute from [x, y, z] to [z, y, x] for SimpleITK
  voxel_arr = permutedims(Float32.(med_image.voxel_data), (3, 2, 1))
  img = sitk.GetImageFromArray(np.array(voxel_arr))
  
  img.SetOrigin(collect(Float64.(med_image.origin)))
  img.SetSpacing(collect(Float64.(med_image.spacing)))
  img.SetDirection(collect(Float64.(med_image.direction)))

  sitk.WriteImage(img, full_path)
end

"""
    update_voxel_data(old_image::MedImage, new_voxel_data::AbstractArray)
"""
function update_voxel_data(old_image::MedImage, new_voxel_data::AbstractArray)
  return @set old_image.voxel_data = new_voxel_data
end

function update_voxel_data(old_image::BatchedMedImage, new_voxel_data::AbstractArray)
  return @set old_image.voxel_data = new_voxel_data
end

"""
    update_voxel_and_spatial_data(old_image, new_voxel_data::AbstractArray, new_origin, new_spacing, new_direction=nothing)
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

Load a medical image from file using SimpleITK via PyCall.
"""
function load_image(path::String, type::String)::MedImage
  sitk = pyimport("SimpleITK")
  
  img = nothing
  if isdir(path)
      # Assume DICOM series
      reader = sitk.ImageSeriesReader()
      dicom_names = reader.GetGDCMSeriesFileNames(path)
      reader.SetFileNames(dicom_names)
      img = reader.Execute()
  else
      img = sitk.ReadImage(path)
  end
  
  # SimpleITK image to array returns [z, y, x]
  voxel_arr_np = sitk.GetArrayFromImage(img)
  # Permute to [x, y, z] for MedImages.jl consistency
  voxel_arr = permutedims(Float32.(voxel_arr_np), (3, 2, 1))
  
  origin = Tuple(Float64.(img.GetOrigin()))
  spacing = Tuple(Float64.(img.GetSpacing()))
  direction = Tuple(Float64.(img.GetDirection()))
  
  study_type = type == "CT" ? MedImage_data_struct.CT_type : MedImage_data_struct.PET_type
  subtype = type == "CT" ? MedImage_data_struct.CT_subtype : MedImage_data_struct.FDG_subtype
  legacy_file_name_field = string(split(path, "/")[length(split(path, "/"))])

  metadata_dict = Dict{Any, Any}()
  try
      metadata_dict = _get_metadata(path)
  catch e
      @warn "Could not extract DICOM metadata using pydicom: $e"
  end

  return MedImage(voxel_data=voxel_arr, origin=origin, spacing=spacing, direction=direction, patient_id="test_id", image_type=study_type, image_subtype=subtype, legacy_file_name=legacy_file_name_field, metadata=metadata_dict)
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
