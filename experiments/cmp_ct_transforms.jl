using MedImages
using MedImages.MedImage_data_struct
using MedImages.Basic_transformations
using MedImages.Spatial_metadata_change
using MedImages.Load_and_save

ct_path = joinpath(@__DIR__, "..", "test_data", "volume-0.nii.gz")
ct = Load_and_save.load_image(ct_path, "CT")
println("CT ", size(ct.voxel_data), " spacing ", ct.spacing)
L = MedImage_data_struct.Linear_en
O = "/tmp/cmp_"

Load_and_save.create_nii_from_medimage(ct, O * "original")

# 1. Rotation 45 deg about axis 3 (degrees)
Load_and_save.create_nii_from_medimage(rotate_mi(ct, 3, 45.0, L), O * "rotate")
println("rotate done")

# 2. Scale 0.5x in-plane
Load_and_save.create_nii_from_medimage(scale_mi(ct, (0.5, 0.5, 1.0), L), O * "scale")
println("scale done")

# 3. Resample to 2mm isotropic
Load_and_save.create_nii_from_medimage(Spatial_metadata_change.resample_to_spacing(ct, (2.0, 2.0, 2.0), L), O * "resample")
println("resample done")

# 4. Crop centred region (0-based beg, size)
Load_and_save.create_nii_from_medimage(crop_mi(ct, (128, 128, 17), (256, 256, 40), L), O * "crop")
println("crop done")

# 5. Pad (beg/end voxels each axis) with air (-1000)
Load_and_save.create_nii_from_medimage(pad_mi(ct, (40, 40, 5), (40, 40, 5), -1000.0, L), O * "pad")
println("pad done")

println("CMP TRANSFORMS DONE")
