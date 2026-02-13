using MedImages
using MedImages.MedImage_data_struct
using MedImages.Utils
using MedImages.Basic_transformations
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
    # Translate by 10 voxels in axis 1
    batch_trans = translate_mi(batch, 10, 1, Linear_en)
    res_trans = unbatch_medimage(batch_trans)
    create_nii_from_medimage(res_trans[1], joinpath(OUTPUT_DIR, "translated_1"))
    create_nii_from_medimage(res_trans[2], joinpath(OUTPUT_DIR, "translated_2"))

    # 4. Crop (Shared)
    println("Applying Batched Cropping (Center Crop to 32x32x32)...")
    # Original 64x64x64. Crop start at 16 (0-based? Basic_transformations uses 0-based for crop_mi arg? Let's check signatures)
    # crop_mi signature: crop_beg::Tuple, crop_size::Tuple
    # 1-based indexing in Julia, but MedImages usually takes 0-based indices for beg?
    # `julia_beg = crop_beg .+ 1` inside crop_mi implies crop_beg is 0-based index.
    # To center crop 32 size from 64 size: start at (64-32)/2 = 16.
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

    println("Done! Check $OUTPUT_DIR for results.")
end

main()
