using MedImages
using MedImages.MedImage_data_struct
using MedImages.Utils
using MedImages.Basic_transformations
using MedImages.Spatial_metadata_change
using MedImages.Resample_to_target
using Dates

# Ensure output directory exists
OUTPUT_DIR = joinpath(@__DIR__, "visual_output")
mkpath(OUTPUT_DIR)
println("Output directory: $OUTPUT_DIR")

function create_synthetic_medimage(size_t::Tuple, type::Symbol)
    data = zeros(Float32, size_t)
    cx, cy, cz = size_t .÷ 2

    if type == :asym_block
        # Create an off-center rectangular block
        # Helps verify orientation and rotation direction
        data[cx-10:cx+10, cy-5:cy+15, cz-10:cz+10] .= 1.0
        # Add a "marker" to distinguish axes
        data[cx+12:cx+15, cy, cz] .= 2.0 # Positive X marker
    elseif type == :sphere
        # Create a sphere
        for z in 1:size_t[3], y in 1:size_t[2], x in 1:size_t[1]
            if (x-cx)^2 + (y-cy)^2 + (z-cz)^2 <= 15^2
                data[x,y,z] = 1.0
            end
        end
    end

    return MedImage(
        voxel_data=data,
        origin=(0.0,0.0,0.0),
        spacing=(1.0,1.0,1.0),
        direction=(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0),
        image_type=MedImages.MedImage_data_struct.MRI_type,
        image_subtype=MedImages.MedImage_data_struct.T1_subtype,
        patient_id=String(type)
    )
end

function main()
    println("Generating synthetic images...")
    img1 = create_synthetic_medimage((64, 64, 64), :asym_block)
    img2 = create_synthetic_medimage((64, 64, 64), :sphere)

    # Save inputs
    create_nii_from_medimage(img1, joinpath(OUTPUT_DIR, "input_1"))
    create_nii_from_medimage(img2, joinpath(OUTPUT_DIR, "input_2"))

    println("Creating batch...")
    batch = create_batched_medimage([img1, img2])

    # 1. Rotate (Unique Angles)
    println("Applying Batched Rotation (Unique: 0 and 45 degrees)...")
    angles = [0.0, 45.0]
    batch_rot = rotate_mi(batch, 3, angles, Linear_en)

    res_rot = unbatch_medimage(batch_rot)
    create_nii_from_medimage(res_rot[1], joinpath(OUTPUT_DIR, "rotated_0deg_1"))
    create_nii_from_medimage(res_rot[2], joinpath(OUTPUT_DIR, "rotated_45deg_2"))

    # 2. Scale (Shared Factor)
    println("Applying Batched Scaling (0.5x)...")
    batch_scale = scale_mi(batch, (0.5, 0.5, 0.5), Linear_en)
    res_scale = unbatch_medimage(batch_scale)
    create_nii_from_medimage(res_scale[1], joinpath(OUTPUT_DIR, "scaled_0.5x_1"))
    create_nii_from_medimage(res_scale[2], joinpath(OUTPUT_DIR, "scaled_0.5x_2"))

    # 3. Translate (Shared)
    println("Applying Batched Translation (10 units X)...")
    batch_trans = translate_mi(batch, 10, 1, Linear_en)
    res_trans = unbatch_medimage(batch_trans)
    create_nii_from_medimage(res_trans[1], joinpath(OUTPUT_DIR, "translated_1"))
    create_nii_from_medimage(res_trans[2], joinpath(OUTPUT_DIR, "translated_2"))

    # 4. Crop (Shared)
    println("Applying Batched Cropping (Center Crop to 32x32x32)...")
    crop_beg = (16, 16, 16)
    crop_size = (32, 32, 32)
    batch_crop = crop_mi(batch, crop_beg, crop_size, Linear_en)
    res_crop = unbatch_medimage(batch_crop)
    create_nii_from_medimage(res_crop[1], joinpath(OUTPUT_DIR, "cropped_1"))
    create_nii_from_medimage(res_crop[2], joinpath(OUTPUT_DIR, "cropped_2"))

    # 5. Pad (Shared)
    println("Applying Batched Padding (5 voxels all sides)...")
    pad_beg = (5, 5, 5)
    pad_end = (5, 5, 5)
    pad_val = 0.0
    batch_pad = pad_mi(batch, pad_beg, pad_end, pad_val, Linear_en)
    res_pad = unbatch_medimage(batch_pad)
    create_nii_from_medimage(res_pad[1], joinpath(OUTPUT_DIR, "padded_1"))
    create_nii_from_medimage(res_pad[2], joinpath(OUTPUT_DIR, "padded_2"))

    # 6. Affine Shearing (Unique)
    println("Applying Batched Affine Shear...")
    # Image 1: No shear
    mat_id = create_affine_matrix()
    # Image 2: Shear X w.r.t Y by 0.5
    mat_shear = create_affine_matrix(shear=(0.5, 0.0, 0.0))
    batch_shear = affine_transform_mi(batch, [mat_id, mat_shear], Linear_en)
    res_shear = unbatch_medimage(batch_shear)
    create_nii_from_medimage(res_shear[1], joinpath(OUTPUT_DIR, "sheared_id_1"))
    create_nii_from_medimage(res_shear[2], joinpath(OUTPUT_DIR, "sheared_xy_0.5_2"))

    # 7. Resample to Spacing (Unique)
    println("Applying Batched Resample to Spacing...")
    # Resample both to 2.0mm spacing. (Original 1.0mm)
    # Output size should be 32x32x32 -> 16x16x16
    spacings = [(2.0, 2.0, 2.0), (2.0, 2.0, 2.0)]
    batch_space = resample_to_spacing(batch, spacings, Linear_en)
    res_space = unbatch_medimage(batch_space)
    create_nii_from_medimage(res_space[1], joinpath(OUTPUT_DIR, "resample_spacing_2mm_1"))
    create_nii_from_medimage(res_space[2], joinpath(OUTPUT_DIR, "resample_spacing_2mm_2"))

    # 8. Resample to Image
    println("Applying Batched Resample to Reference...")
    # Create a reference batch with smaller field of view (e.g. 32x32x32) and offset origin
    ref_img1 = MedImage(voxel_data=zeros(Float32, 32, 32, 32), origin=(10.0,10.0,10.0), spacing=(1.0,1.0,1.0), direction=img1.direction, image_type=MedImages.MedImage_data_struct.MRI_type, image_subtype=MedImages.MedImage_data_struct.T1_subtype, patient_id="ref1")
    ref_img2 = MedImage(voxel_data=zeros(Float32, 32, 32, 32), origin=(-10.0,-10.0,-10.0), spacing=(1.0,1.0,1.0), direction=img2.direction, image_type=MedImages.MedImage_data_struct.MRI_type, image_subtype=MedImages.MedImage_data_struct.T1_subtype, patient_id="ref2")
    ref_batch = create_batched_medimage([ref_img1, ref_img2])

    batch_res_img = resample_to_image(ref_batch, batch, Linear_en)
    res_img = unbatch_medimage(batch_res_img)
    create_nii_from_medimage(res_img[1], joinpath(OUTPUT_DIR, "resample_to_img_1"))
    create_nii_from_medimage(res_img[2], joinpath(OUTPUT_DIR, "resample_to_img_2"))

    println("Done! Check $OUTPUT_DIR for results.")
end

main()
