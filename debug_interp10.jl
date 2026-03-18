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

for ix in 1:64
    real_x = (Float32(ix) - 1.0f0) * (Float32(new_spacing[1]) / Float32(old_spacing[1])) + 1.0f0
    x0 = floor(Int, real_x)
    x1 = min(x0 + 1, size_src[1])
    xd = real_x - x0
    val_fast = real_x > 32.0f0 ? 0.0f0 : data[x0, 1, 1] * (1-xd) + data[x1, 1, 1] * xd
    val_slow = itp(real_x, 1.0, 1.0)
    if abs(val_fast - val_slow) > 1e-4
        println("Diff at $ix (x=$real_x): Fast=$val_fast, Slow=$val_slow")
    end
end
