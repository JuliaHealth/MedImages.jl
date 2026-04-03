using Pkg
Pkg.activate(".")
using MedImages
using Statistics

println("--- Intensity Normalization Verification ---")
println("Loading sample image...")
mi = load_image("test_data/volume-0.nii.gz", "CT")

println("Original Stats:")
println("  Mean: ", mean(mi.voxel_data))
println("  Std:  ", std(mi.voxel_data))
println("  Min:  ", minimum(mi.voxel_data))
println("  Max:  ", maximum(mi.voxel_data))

println("\nApplying Z-score Normalization...")
norm = z_score_normalize(mi)

println("Normalized Stats:")
println("  Mean: ", mean(norm.voxel_data))
println("  Std:  ", std(norm.voxel_data))
println("  Min:  ", minimum(norm.voxel_data))
println("  Max:  ", maximum(norm.voxel_data))

println("\n✓ If Mean is ~0 and Std is ~1, normalization is correct!")
