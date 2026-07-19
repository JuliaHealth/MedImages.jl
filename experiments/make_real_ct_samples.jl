# Apply MedImages.jl spatial operations to a real CT volume and save the
# results as NIfTI, so the figures in the README can be reproduced end-to-end.
#
#   julia +1.11 --startup-file=no --project=. experiments/make_real_ct_samples.jl
#
# Outputs land in test/visual_output/real_ct/ and are consumed by
# experiments/render_real_ct_grid.py.

using MedImages
using MedImages.MedImage_data_struct
using MedImages.Basic_transformations
# resample_to_spacing is qualified below (MedImages.resample_to_spacing) to avoid
# the name clash between `using MedImages` and the Resample_to_target submodule.

const CT_PATH = joinpath(@__DIR__, "..", "test_data", "volume-0.nii.gz")
const OUT_DIR = joinpath(@__DIR__, "..", "test", "visual_output", "real_ct")
const AIR = -1000.0   # CT background (Hounsfield air) for padding / extrapolation

mkpath(OUT_DIR)
save(img, name) = create_nii_from_medimage(img, joinpath(OUT_DIR, name))

function main()
    println("Loading real CT: $CT_PATH")
    ct = load_image(CT_PATH, "CT")
    println("  size = $(size(ct.voxel_data)), spacing = $(ct.spacing)")
    save(ct, "ct_original")

    println("Rotate 45deg about Z...")
    save(rotate_mi(ct, 3, 45.0, Linear_en), "ct_rotate45")

    println("Scale 0.5x (in-plane; axial slices keep their count)...")
    save(scale_mi(ct, (0.5, 0.5, 1.0), Linear_en), "ct_scale05")

    println("Resample to 2mm isotropic...")
    save(MedImages.resample_to_spacing(ct, (2.0, 2.0, 2.0), Linear_en), "ct_resample2mm")

    println("Crop to 256x256x40 (centered)...")
    save(crop_mi(ct, (128, 128, 17), (256, 256, 40), Linear_en), "ct_crop")

    println("Pad +40 (in-plane) / +5 (axial)...")
    save(pad_mi(ct, (40, 40, 5), (40, 40, 5), AIR, Linear_en), "ct_pad")

    println("Fused affine (rotate 30deg Z + scale 0.8x + translate +5/-2)...")
    mat = create_affine_matrix(
        rotation    = (0.0, 0.0, 30.0),
        scale       = (0.8, 0.8, 0.8),
        translation = (5.0, -2.0, 0.0),
    )
    save(affine_transform_mi(ct, mat, Linear_en), "ct_fused")

    println("Done -> $OUT_DIR")
end

main()
