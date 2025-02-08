
module Load_and_save
using Dictionaries, Dates, PyCall
using Accessors, UUIDs, ITKIOWrapper
using ..MedImage_data_struct
using ..MedImage_data_struct:MedImage
using ..Brute_force_orientation
using ..Utils
export load_images
export load_image
export update_voxel_and_spatial_data

"""
helper function for dicom #1
returns an array of unique SERIES INSTANCE UID within dicom files within a dicom directory
"""
function unique_series_id_within_dicom_files(dicom_data_array)
  return " "
  # return map(dicom_file_data -> dicom_file_data[tag"SeriesInstanceUID"], dicom_data_array) |>
  #        Set |>
  #        collect
end

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
helper function for nifti #1
Determines the study type for an image based on its metadata using SimpleITK
"""
function infer_modality(image)
    sitk = pyimport("SimpleITK")
    metadata = Dict()
    for key in image.GetMetaDataKeys()
        metadata[key] = image.GetMetaData(key)
    end

    # Check for CT-specific metadata
    if "modality" in keys(metadata) && metadata["modality"] == "CT"
        return "CT"
    end

    spacing = image.GetSpacing()
    size = image.GetSize()
    pixel_type = image.GetPixelIDTypeAsString()

    # Relaxed spacing condition for CT
    spacing_condition_ct = all(s -> s <= 1.0, spacing)
    pixel_type_condition_ct = pixel_type == "16-bit signed integer"

    if spacing_condition_ct && pixel_type_condition_ct
        # Additional check: sample some voxels to see if they're in the typical CT range
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        mean_value = stats.GetMean()

        if -1000 <= mean_value <= 3000  # Typical range for CT in Hounsfield units
            return MedImage_data_struct.CT_type
        end
    end

    # PET condition (unchanged)
    spacing_condition_pet = all(s -> s > 1.0, spacing)
    pixel_type_condition_pet = occursin("float", lowercase(pixel_type))

    if spacing_condition_pet && pixel_type_condition_pet
        return MedImage_data_struct.PET_type
    end

    return MedImage_data_struct.CT_type
end


function load_images(path::String)::Array{MedImage}
  if isdir(path)
    dicom_data_array = DICOM.dcmdir_parse(path)
    return unique_series_id_within_dicom_files(dicom_data_array) |>
           series_ids -> map(series_id -> filter(dcm -> dcm.SeriesInstanceUID == series_id, dicom_data_array), series_ids) |> #now we have series of dicom files with the same series id
                         dcom_data_locs -> map(dcom_data_loc -> MedImage([get_pixel_data(dcom_data_loc),
        dcom_data_loc[1].PixelSpacing,
        dcom_data_loc[1].ImageOrientationPatient,
        dcom_data_loc[1].ImagePositionPatient,
        " ",#dcom_data_loc[1].StudyCompletionDate
        dcom_data_loc[1].PatientID]), dcom_data_locs)
  else

    #defaulting to LPS orientation for all images
    Brute_force_orientation.change_image_orientation(path, MedImage_data_struct.ORIENTATION_LPS)
    spatial_meta = ITKIOWrapper.loadSpatialMetaData(path)

    #necessary for inferring modality
    sitk = pyimport("SimpleITK")
    itk_nifti_image = sitk.ReadImage(path)



    origin = spatial_meta.origin
    # origin = set_origin_for_nifti_file(sform_qform_similar, nifti_image_struct.sto_xyz)
    spacing = spatial_meta.spacing  #set_spacing_for_nifti_files([nifti_image_struct.dx, nifti_image_struct.dy,nifti_image_struct.dz])
    # spacing=(spacing[3],spacing[2],spacing[1])
    # direction = set_direction_for_nifti_file(path, header_data_dict["qform_code_name"], header_data_dict["sform_code_name"], sform_qform_similar)
    direction = spatial_meta.direction
    voxel_arr = ITKIOWrapper.loadVoxelData(path,spatial_meta)
    voxel_arr = voxel_arr.dat
    voxel_arr = permutedims(voxel_arr, (3, 2, 1))
    spatial_metadata_keys = ["origin", "spacing", "direction"]
    spatial_metadata_values = [origin, spacing, direction]
    spatial_metadata = Dictionaries.Dictionary(spatial_metadata_keys, spatial_metadata_values)
    legacy_file_name_field = string(split(path, "/")[length(split(path, "/"))])

    return [MedImage(voxel_data=voxel_arr, origin=origin, spacing=spacing, direction=direction, patient_id="test_id", image_type=infer_modality(itk_nifti_image), image_subtype=MedImage_data_struct.CT_subtype, legacy_file_name=legacy_file_name_field)]


  end

end



function create_nii_from_medimage(med_image::MedImage, file_path::String)
  # Convert voxel_data to a numpy array (Assuming voxel_data is stored in Julia array format)
  voxel_data_np = med_image.voxel_data
  voxel_data_np = permutedims(voxel_data_np, (3, 2, 1))
  # Create a SimpleITK image from numpy array
  sitk = pyimport("SimpleITK")
  image_sitk = sitk.GetImageFromArray(voxel_data_np)

  # Set spatial metadata
  image_sitk.SetOrigin(med_image.origin)
  image_sitk.SetSpacing(med_image.spacing)
  image_sitk.SetDirection(med_image.direction)

  # Save the image as .nii.gz
  sitk.WriteImage(image_sitk, file_path * ".nii.gz")
end



function update_voxel_data(old_image, new_voxel_data::AbstractArray)

  return MedImage(
    new_voxel_data,
    old_image.origin,
    old_image.spacing,
    old_image.direction,
    instances(Image_type)[Int(old_image.image_type)+1],#  Int(image.image_type),
    instances(Image_subtype)[Int(old_image.image_subtype)+1],#  Int(image.image_subtype),
    # old_image.voxel_datatype,
    old_image.date_of_saving,
    old_image.acquistion_time,
    old_image.patient_id,
    instances(current_device_enum)[Int(old_image.current_device)+1], #Int(image.current_device),
    old_image.study_uid,
    old_image.patient_uid,
    old_image.series_uid,
    old_image.study_description,
    old_image.legacy_file_name,
    old_image.display_data,
    old_image.clinical_data,
    old_image.is_contrast_administered,
    old_image.metadata)

end

function update_voxel_and_spatial_data(old_image, new_voxel_data::AbstractArray
  , new_origin, new_spacing, new_direction)

  return MedImage(
    new_voxel_data,
    new_origin,
    new_spacing,
    new_direction,
    # old_image.spatial_metadata,
    instances(Image_type)[Int(old_image.image_type)+1],#  Int(image.image_type),
    instances(Image_subtype)[Int(old_image.image_subtype)+1],#  Int(image.image_subtype),
    # old_image.voxel_datatype,
    old_image.date_of_saving,
    old_image.acquistion_time,
    old_image.patient_id,
    instances(current_device_enum)[Int(old_image.current_device)+1], #Int(image.current_device),
    old_image.study_uid,
    old_image.patient_uid,
    old_image.series_uid,
    old_image.study_description,
    old_image.legacy_file_name,
    old_image.display_data,
    old_image.clinical_data,
    old_image.is_contrast_administered,
    old_image.metadata)

end

# function update_voxel_and_spatial_data(old_image, new_voxel_data::AbstractArray, new_origin, new_spacing, new_direction)

#   res = @set old_image.voxel_data = new_voxel_data
#   res = @set res.origin = Utils.ensure_tuple(new_origin)
#   res = @set res.spacing = Utils.ensure_tuple(new_spacing)
#   res = @set res.direction = Utils.ensure_tuple(new_direction)
#   # voxel_data=new_voxel_data
#   # origin=new_origin
#   # spacing=new_spacing
#   # direction=new_direction

#   # return @pack! old_image = voxel_data, origin, spacing, direction
#   return res
# end

"""
load image from path
"""
function load_image(path)
  # test_image_equality(p,p)

  medimage_instance_array = load_images(path)
  medimage_instance = medimage_instance_array[1]
  return medimage_instance
end#load_image

end




