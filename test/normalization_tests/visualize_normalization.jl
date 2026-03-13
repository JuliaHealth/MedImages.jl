using Pkg
Pkg.activate(".")
using MedImages
using Statistics
using PyCall

# Ensure SimpleITK is available
sitk = pyimport("SimpleITK")

println("--- Visual Proof Generation ---")
println("Loading sample image...")
mi = load_image("test_data/volume-0.nii.gz", "CT")

# Normalization
println("Applying normalization...")
norm = z_score_normalize(mi)
# Rescale to 0-255 for PNG saving
min_val = minimum(norm.voxel_data)
max_val = maximum(norm.voxel_data)
norm_0_255 = (norm.voxel_data .- min_val) ./ (max_val - min_val) .* 255.0

# Extract a middle slice
dims = size(mi.voxel_data)
slice_idx = dims[3] ÷ 2
original_slice = mi.voxel_data[:, :, slice_idx]
normalized_slice = norm_0_255[:, :, slice_idx]

# Convert to sitk image and save as PNG
# Note: we need to cast to uint8 for PNG
function save_as_png(data, filename)
    # Permute if necessary for sitk (y, x) -> (x, y)
    # SimpleITK expects [x, y]
    sitk_img = sitk.GetImageFromArray(collect(data'))
    sitk_img = sitk.Cast(sitk_img, sitk.sitkUInt8)
    sitk.WriteImage(sitk_img, filename)
    println("Saved: $filename")
end

# Normalize original slice to 0-255 for visualization
orig_min = minimum(original_slice)
orig_max = maximum(original_slice)
original_slice_0_255 = (original_slice .- orig_min) ./ (orig_max - orig_min) .* 255.0

save_as_png(original_slice_0_255, "original_slice.png")
save_as_png(normalized_slice, "transformed_slice.png")

println("\n✓ Visualization generated successfully!")
