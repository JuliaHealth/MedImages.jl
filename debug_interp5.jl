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

for extrapolate_value in [0.0, NaN]
    itp = extrapolate(scale(interpolate(data, BSpline(Linear())), A_x1, A_x1, A_x1), extrapolate_value)
    println("Extrapolate ", extrapolate_value, " at 32.5: ", itp(32.5, 1.0, 1.0))
end
