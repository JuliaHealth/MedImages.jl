using KernelAbstractions
using Interpolations
using Test
using MedImages
using MedImages.Utils

size_src = (32, 32, 32)
data = rand(Float32, size_src)
old_spacing = (1.0, 1.0, 1.0)
new_spacing = (0.5, 0.5, 0.5)

new_dims = Tuple(Int(ceil(sz * osp / nsp)) for (sz, osp, nsp) in zip(size_src, old_spacing, new_spacing))

# 1. Fast Kernel (Optimized)
res_fast = Utils.resample_kernel_launch(data, old_spacing, new_spacing, new_dims, MedImages.Linear_en)

# 2. Slow Interpolations.jl (Reference)
indices = Utils.get_base_indicies_arr(new_dims)
points = similar(indices, Float64)
for i in 1:size(indices, 2)
    points[1, i] = (indices[1, i] - 1) * new_spacing[1] + 1.0
    points[2, i] = (indices[2, i] - 1) * new_spacing[2] + 1.0
    points[3, i] = (indices[3, i] - 1) * new_spacing[3] + 1.0
end

res_slow_flat = Utils.interpolate_my(points, data, old_spacing, MedImages.Linear_en, true, 0.0, false)
res_slow = reshape(res_slow_flat, new_dims)
res_slow = Float32.(res_slow)

diff = abs.(res_fast .- res_slow)
println("Mean diff: ", sum(diff)/length(diff))
println("Max diff: ", maximum(diff))

# Let's inspect the point with the max difference
idx = argmax(diff)
i, j, k = Tuple(CartesianIndices(new_dims)[idx])
println("Max diff at index: ", (i, j, k))
println("Fast val: ", res_fast[i, j, k])
println("Slow val: ", res_slow[i, j, k])
println("Source shape: ", size_src)

println("point for max diff: ", points[:, idx])
