using MedImages
using MedImages.MedImage_data_struct
using MedImages.Basic_transformations
using MedImages.Spatial_metadata_change
using MedImages.Load_and_save

const TEST_DATA = joinpath(@__DIR__, "..", "test_data")
const OUT = "/tmp/real_ct"

ct_path = joinpath(TEST_DATA, "volume-0.nii.gz")
isfile(ct_path) || error("missing $ct_path")

println("Loading real CT: $ct_path")
ct = Load_and_save.load_image(ct_path, "CT")
println("Loaded size: ", size(ct.voxel_data), " spacing: ", ct.spacing)

Load_and_save.create_nii_from_medimage(ct, OUT * "_original")
println("saved original")

# NOTE: rotate_mi expects the angle in DEGREES (Rodrigues_rotation_matrix applies deg2rad
# internally), despite the docstring saying "radians". Passing degrees here.
ct_rot = rotate_mi(ct, 3, 45.0, MedImage_data_struct.Linear_en)
Load_and_save.create_nii_from_medimage(ct_rot, OUT * "_rotated_45")
println("saved rotated 45deg, size: ", size(ct_rot.voxel_data))

ct_scaled = scale_mi(ct, (0.5, 0.5, 1.0), MedImage_data_struct.Linear_en)
Load_and_save.create_nii_from_medimage(ct_scaled, OUT * "_scaled_half")
println("saved scaled 0.5x, size: ", size(ct_scaled.voxel_data))

ct_res = Spatial_metadata_change.resample_to_spacing(ct, (2.0, 2.0, 2.0), MedImage_data_struct.Linear_en)
Load_and_save.create_nii_from_medimage(ct_res, OUT * "_resampled_2mm")
println("saved resampled 2mm, size: ", size(ct_res.voxel_data))

println("CT TRANSFORMS DONE")
