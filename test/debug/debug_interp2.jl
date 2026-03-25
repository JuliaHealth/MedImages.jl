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

# Check point 32.5
println("Interpolations.jl at 32.5: ", itp(32.5, 1.0, 1.0))

# Now check how Fast Kernel handles 32.5
println("Fast kernel logic at 32.5:")
sx = 32
real_x = 32.5f0
if real_x < 1.0f0 || real_x > Float32(sx)
    println("Out of bounds")
else
    x0 = floor(Int, real_x)
    x1 = min(x0 + 1, sx)
    xd = real_x - x0
    println("x0: ", x0, " x1: ", x1, " xd: ", xd)
    # it interpolates between data[32] and data[32]! Because x1 is clamped to 32.
end
