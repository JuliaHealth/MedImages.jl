module MedImages

# Export submodules
export Brute_force_orientation
export MedImage_data_struct
export Load_and_save
export Basic_transformations
export Resample_to_target
export Spatial_metadata_change
export Utils
export SUV_calc
export Normalization

# Export key functions that tests need
export load_image, update_voxel_data, update_voxel_and_spatial_data, create_nii_from_medimage
export calculate_suv_factor, calculate_suv_statistics
export save_med_image, load_med_image
export resample_to_spacing, change_orientation
export resample_to_image, rotate_mi, crop_mi, pad_mi, translate_mi, scale_mi, pad_or_crop_mi
export one_hot_encode, calculate_barycenter, calculate_max_radius, extract_points_from_mask
export Rodrigues_rotation_matrix, create_affine_matrix, compose_affine_matrices, affine_transform_mi
export string_to_orientation_enum, orientation_enum_to_string
export z_score_normalize, min_max_normalize, apply_dicom_rescale, histogram_match, nyul_train, nyul_transform
export create_batched_medimage, unbatch_medimage

# Export enums and types
export MedImage, BatchedMedImage, Image_type, Image_subtype, current_device_enum
export Interpolator_enum, Mode_mi, Orientation_code
export Nearest_neighbour_en, Linear_en, B_spline_en
# Export orientation enum values
export ORIENTATION_RPI, ORIENTATION_LPI, ORIENTATION_RAI, ORIENTATION_LAI
export ORIENTATION_RPS, ORIENTATION_LPS, ORIENTATION_RAS, ORIENTATION_LAS
export MRI_type, PET_type, CT_type
export CT_subtype, ADC_subtype, DWI_subtype, T1_subtype, T2_subtype, FLAIR_subtype, FDG_subtype, PSMA_subtype
export CPU_current_device, CUDA_current_device, AMD_current_device, ONEAPI_current_device

include("MedImage_data_struct.jl")
include("Orientation_dicts.jl")
include("Brute_force_orientation.jl")
include("Utils.jl")
include("Load_and_save.jl")
include("Basic_transformations.jl")
include("Spatial_metadata_change.jl")
include("Resample_to_target.jl")
include("HDF5_manag.jl")
include("SUV.jl")
include("Normalization.jl")

# Re-export functions from submodules
using .Utils
using .SUV_calc: calculate_suv_factor, calculate_suv_statistics
using .Load_and_save: load_image, update_voxel_data, update_voxel_and_spatial_data, create_nii_from_medimage
using .MedImage_data_struct: MedImage, BatchedMedImage, Image_type, Image_subtype, current_device_enum
using .MedImage_data_struct: Interpolator_enum, Mode_mi, Orientation_code
using .MedImage_data_struct: Nearest_neighbour_en, Linear_en, B_spline_en
using .MedImage_data_struct: ORIENTATION_RPI, ORIENTATION_LPI, ORIENTATION_RAI, ORIENTATION_LAI
using .MedImage_data_struct: ORIENTATION_RPS, ORIENTATION_LPS, ORIENTATION_RAS, ORIENTATION_LAS
using .MedImage_data_struct: MRI_type, PET_type, CT_type
using .MedImage_data_struct: CT_subtype, ADC_subtype, DWI_subtype, T1_subtype, T2_subtype, FLAIR_subtype, FDG_subtype, PSMA_subtype
using .MedImage_data_struct: CPU_current_device, CUDA_current_device, AMD_current_device, ONEAPI_current_device
using .Orientation_dicts: string_to_orientation_enum, orientation_enum_to_string
using .Spatial_metadata_change: change_orientation
using .Resample_to_target: resample_to_image, resample_to_spacing
using .Basic_transformations: rotate_mi, crop_mi, pad_mi, translate_mi, scale_mi, pad_or_crop_mi
using .Basic_transformations: Rodrigues_rotation_matrix, create_affine_matrix, compose_affine_matrices, affine_transform_mi
using .Normalization: z_score_normalize, min_max_normalize, apply_dicom_rescale, histogram_match, nyul_train, nyul_transform

# Make HDF5 functions available (they're not in a module)
export save_med_image, load_med_image

end #MedImages