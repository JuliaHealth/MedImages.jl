using MedImages
using MedImages.MedImage_data_struct
using MedImages.Utils
using MedImages.Basic_transformations
using MedImages.Load_and_save
using MedImages.Resample_to_target
using MedImages.Spatial_metadata_change
using MedImages.Orientation_dicts
using Statistics

# Config
BASE_DIR = "/home/jm/project_ssd/MedImages.jl/test/visual_output/local"
STEP1_DIR = joinpath(BASE_DIR, "step1_resampled")
STEP2_DIR = joinpath(BASE_DIR, "step2_batched_rot_trans")
STEP3_DIR = joinpath(BASE_DIR, "step3_unique_affine")

mkpath(STEP1_DIR)
mkpath(STEP2_DIR)
mkpath(STEP3_DIR)

# 1. Load images
println("Loading images...")
ct_ref_path = joinpath(BASE_DIR, "CT.nii.gz")
dose_path = joinpath(BASE_DIR, "Dosemap.nii.gz")
nm_path = joinpath(BASE_DIR, "NM_Vendor.nii.gz")
spect_path = joinpath(BASE_DIR, "SPECT_Recon_WholeBody.nii.gz")

# Load_and_save.load_image requires "CT" or "PET" to set modality
# We use CT for the reference and PET/CT for others to ensure basic metadata set
ct = load_image(ct_ref_path, "CT")
dose = load_image(dose_path, "PET")
nm = load_image(nm_path, "CT")
spect = load_image(spect_path, "PET")

# Step 1: Resample all to CT.nii.gz
println("\nStep 1: Resampling to CT reference geometry...")
# Note: CT is already in CT geometry, but we can resample it to itself to be safe or just use it.
ct_res = ct 
dose_res = resample_to_image(ct, dose, Linear_en)
nm_res = resample_to_image(ct, nm, Linear_en)
spect_res = resample_to_image(ct, spect, Linear_en)

# Save Step 1
println("Saving Step 1 results...")
create_nii_from_medimage(ct_res, joinpath(STEP1_DIR, "CT_resampled"))
create_nii_from_medimage(dose_res, joinpath(STEP1_DIR, "Dosemap_resampled"))
create_nii_from_medimage(nm_res, joinpath(STEP1_DIR, "NM_Vendor_resampled"))
create_nii_from_medimage(spect_res, joinpath(STEP1_DIR, "SPECT_Recon_resampled"))

# Step 2: Rotate all 90 degrees and translate 5 cm down (batched)
println("\nStep 2: Batched 90-deg rotation + 5cm translation...")
# Create batch from Step 1 results
batch = create_batched_medimage([ct_res, dose_res, nm_res, spect_res])

# Translation 5cm down. 
# Assuming spacing in mm, 5cm = 50.0 units.
# Axial images: Z (axis 3) is Inferior-Superior. "Down" (Inferior) is negative Z.
# BUT wait, orientation matters. If it's LPS, Z- is inferior.
# Let's check spacing to ensure we are in mm.
println("CT Spacing: $(ct.spacing)")
trans_val = -50.0 # 5cm down in Z

# Batched Affine (Shared for all)
mat_step2 = create_affine_matrix(rotation=(0.0, 0.0, 90.0), translation=(0.0, 0.0, trans_val))
batch_step2 = affine_transform_mi(batch, mat_step2, Linear_en)

# Save Step 2
println("Saving Step 2 results...")
res_step2 = unbatch_medimage(batch_step2)
create_nii_from_medimage(res_step2[1], joinpath(STEP2_DIR, "CT_rot_trans"))
create_nii_from_medimage(res_step2[2], joinpath(STEP2_DIR, "Dosemap_rot_trans"))
create_nii_from_medimage(res_step2[3], joinpath(STEP2_DIR, "NM_Vendor_rot_trans"))
create_nii_from_medimage(res_step2[4], joinpath(STEP2_DIR, "SPECT_rot_trans"))

# Step 3: Rotate each image and shear by different random values (batched)
println("\nStep 3: Batched unique random rotation + shear...")
# Unique matrices for the batch
mats_step3 = [
    create_affine_matrix(rotation=(10.0, 5.0, 0.0), shear=(0.1, 0.0, 0.0)),
    create_affine_matrix(rotation=(0.0, 10.0, 5.0), shear=(0.0, 0.1, 0.0)),
    create_affine_matrix(rotation=(5.0, 0.0, 10.0), shear=(0.0, 0.0, 0.1)),
    create_affine_matrix(rotation=(15.0, 15.0, 15.0), shear=(0.1, 0.1, 0.1))
]
batch_step3 = affine_transform_mi(batch_step2, mats_step3, Linear_en)

# Save Step 3
println("Saving Step 3 results...")
res_step3 = unbatch_medimage(batch_step3)
create_nii_from_medimage(res_step3[1], joinpath(STEP3_DIR, "CT_random"))
create_nii_from_medimage(res_step3[2], joinpath(STEP3_DIR, "Dosemap_random"))
create_nii_from_medimage(res_step3[3], joinpath(STEP3_DIR, "NM_Vendor_random"))
create_nii_from_medimage(res_step3[4], joinpath(STEP3_DIR, "SPECT_random"))

println("\nVisual Verification Test Complete! Results saved in $BASE_DIR")
