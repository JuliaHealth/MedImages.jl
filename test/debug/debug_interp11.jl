using KernelAbstractions
using Interpolations
using Test
using MedImages
using MedImages.Utils
using Statistics

size_src = (32, 32, 32)
data = rand(Float32, size_src)
old_spacing = (1.0, 1.0, 1.0)
new_spacing = (0.5, 0.5, 0.5)

new_dims = Tuple(Int(ceil(sz * osp / nsp)) for (sz, osp, nsp) in zip(size_src, old_spacing, new_spacing))

# Fast uses extrapolate_value = median of corners. Let's see what that is.
corners = Utils.extract_corners(data)
extrap_val = median(corners)
println("Fast kernel extrap_val = ", extrap_val)

# Wait, the test uses:
# res_slow_flat = Utils.interpolate_my(points, data, old_spacing, MedImages.Linear_en, true, 0.0, false)
# which explicitly passes `0.0` as extrapolate_value!

# So Fast uses median(corners), Slow uses 0.0! That's the difference!
