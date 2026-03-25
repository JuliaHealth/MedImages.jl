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

A_x1 = 1:old_spacing[1]:(1+old_spacing[1]*(size_src[1]-1))
itp = extrapolate(scale(interpolate(data, BSpline(Linear())), A_x1, A_x1, A_x1), 0.0)

for ix in 63:64
    real_x = (Float32(ix) - 1.0f0) * (Float32(new_spacing[1]) / Float32(old_spacing[1])) + 1.0f0
    println("ix=$ix -> real_x=$real_x, itp=$(itp(real_x, 1.0, 1.0))")
end

println("Float64 logic:")
for ix in 63:64
    real_x = (Float64(ix) - 1.0) * (Float64(new_spacing[1]) / Float64(old_spacing[1])) + 1.0
    println("ix=$ix -> real_x=$real_x, itp=$(itp(real_x, 1.0, 1.0))")
end
