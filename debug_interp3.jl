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

# Check point 32.0, 31.5, 32.5
println("Interpolations.jl at 32.0: ", itp(32.0, 1.0, 1.0))
println("Interpolations.jl at 31.5: ", itp(31.5, 1.0, 1.0))
println("Interpolations.jl at 32.5: ", itp(32.5, 1.0, 1.0))

# Now check how Fast Kernel logic handles this for new_dims points
for ix in 63:64
    real_x = (Float32(ix) - 1.0f0) * (Float32(new_spacing[1]) / Float32(old_spacing[1])) + 1.0f0
    println("ix=$ix -> real_x = $real_x")
    if real_x < 1.0f0 || real_x > Float32(size_src[1])
        println("Fast: Out of bounds")
    else
        x0 = floor(Int, real_x)
        x1 = min(x0 + 1, size_src[1])
        xd = real_x - x0
        val = data[x0, 1, 1] * (1-xd) + data[x1, 1, 1] * xd
        println("Fast: $val (x0=$x0, x1=$x1, xd=$xd)")
    end
end
