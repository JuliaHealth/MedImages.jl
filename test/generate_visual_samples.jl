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
    # Image 1: 0 degrees (should be unchanged)
    # Image 2: 45 degrees around Z (axis 3)
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

    println("Done! Check $OUTPUT_DIR for results.")
end

main()
