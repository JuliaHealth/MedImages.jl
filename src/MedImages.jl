module MedImages

# Export submodules
export Brute_force_orientation
export MedImage_data_struct
export Load_and_save
export Basic_transformations
export Resample_to_target
export Spatial_metadata_change
export Utils

# Export key functions that tests need
export load_image, update_voxel_and_spatial_data, create_nii_from_medimage
export save_med_image, load_med_image
export resample_to_spacing, change_orientation
export resample_to_image, rotate_mi, crop_mi, pad_mi, translate_mi, scale_mi
export string_to_orientation_enum, orientation_enum_to_string

# Export enums and types
export MedImage, Image_type, Image_subtype, current_device_enum
export Interpolator_enum, Mode_mi, Orientation_code
export Nearest_neighbour_en, Linear_en, B_spline_en

include("MedImage_data_struct.jl")
include("Orientation_dicts.jl")
include("Brute_force_orientation.jl")
include("Utils.jl")
include("Load_and_save.jl")
include("Basic_transformations.jl")
include("Spatial_metadata_change.jl")
include("Resample_to_target.jl")
include("HDF5_manag.jl")

# Re-export functions from submodules
using .Utils
using .Load_and_save: load_image, update_voxel_and_spatial_data, create_nii_from_medimage
using .MedImage_data_struct: MedImage, Image_type, Image_subtype, current_device_enum
using .MedImage_data_struct: Interpolator_enum, Mode_mi, Orientation_code
using .MedImage_data_struct: Nearest_neighbour_en, Linear_en, B_spline_en
using .Orientation_dicts: string_to_orientation_enum, orientation_enum_to_string
using .Spatial_metadata_change: resample_to_spacing, change_orientation
using .Resample_to_target: resample_to_image
using .Basic_transformations: rotate_mi, crop_mi, pad_mi, translate_mi, scale_mi

# Make HDF5 functions available (they're not in a module)
export save_med_image, load_med_image

end #MedImages