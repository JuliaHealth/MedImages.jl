module MedImages
export MedImage_data_struct
export Load_and_save
export Basic_transformations
export Resample_to_target
export Spatial_metadata_change
export Utils


include("MedImage_data_struct.jl")
include("Load_and_save.jl")
include("Utils.jl")
include("Basic_transformations.jl")
include("Resample_to_target.jl")
include("Spatial_metadata_change.jl")
# include("Orientation_dicts.jl")
# include("Brute_force_orientation.jl")
include("HDF5_manag.jl")
end #MedImages
