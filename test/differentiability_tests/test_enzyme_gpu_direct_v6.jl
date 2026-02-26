using Enzyme
using KernelAbstractions
using CUDA
using Test

if !CUDA.functional()
    exit(0)
end

# Pure scalar logic for one point
@inline function point_wise_scalar(
    m11, m12, m13, m14,
    m21, m22, m23, m24,
    m31, m32, m33, m34,
    v000, v100, v010, v110, v001, v101, v011, v111,
    ix, iy, iz, shift_x, shift_y, shift_z,
    src_sh_x, src_sh_y, src_sh_z
)
    # 1. Transform
    px = Float32(ix) - shift_x
    py = Float32(iy) - shift_y
    pz = Float32(iz) - shift_z
    
    real_x = m11*px + m12*py + m13*pz + m14 + shift_x
    real_y = m21*px + m22*py + m23*pz + m24 + shift_y
    real_z = m31*px + m32*py + m33*pz + m34 + shift_z
    
    # 2. Bounds
    if real_x < 1.0f0 || real_y < 1.0f0 || real_z < 1.0f0 || real_x > src_sh_x || real_y > src_sh_y || real_z > src_sh_z
        return 0.0f0
    end
    
    # 3. Trilinear Interpolation weights
    x0 = floor(real_x)
    y0 = floor(real_y)
    z0 = floor(real_z)
    
    xd = real_x - x0
    yd = real_y - y0
    zd = real_z - z0
    
    # 4. Interpolate using the 8 values passed
    c00 = v000 * (1.0f0 - xd) + v100 * xd
    c10 = v010 * (1.0f0 - xd) + v110 * xd
    c01 = v001 * (1.0f0 - xd) + v101 * xd
    c11 = v011 * (1.0f0 - xd) + v111 * xd

    c0 = c00 * (1.0f0 - yd) + c10 * yd
    c1 = c01 * (1.0f0 - yd) + c11 * yd

    return c0 * (1.0f0 - zd) + c1 * zd
end

@kernel function grad_kernel_v6(d_src, d_mat, @Const(d_out), @Const(src), @Const(mat), @Const(src_sh), @Const(out_sh), @Const(shift))
    I = @index(Global)
    
    n_spatial = out_sh[1] * out_sh[2] * out_sh[3]
    idx_spatial = (I - 1) % n_spatial + 1
    idx_batch = (I - 1) ÷ n_spatial + 1
    
    # ... coordinate mapping ...
    # (Leaving out for brevity in test, just use I)
    ix, iy, iz = 1,1,1 # Placeholder
    
    mat_idx = size(mat,3) == 1 ? 1 : idx_batch
    
    # Pre-fetch neighbors (Const)
    # The actual indices would depend on real_x/y/z which depend on mat
    # This is tricky because we need gradients w.r.t the SAMPLING indices too
    # BUT wait, the values v000... are constants for the local AD.
    # The dependency on mat is through xd, yd, zd.
    
    # Let's see if this compiles
    res = Enzyme.autodiff_deferred(
        Reverse,
        point_wise_scalar,
        Active,
        Active(mat[1,1,mat_idx]), Active(mat[1,2,mat_idx]), Active(mat[1,3,mat_idx]), Active(mat[1,4,mat_idx]),
        Active(mat[2,1,mat_idx]), Active(mat[2,2,mat_idx]), Active(mat[2,3,mat_idx]), Active(mat[2,4,mat_idx]),
        Active(mat[3,1,mat_idx]), Active(mat[3,2,mat_idx]), Active(mat[3,3,mat_idx]), Active(mat[3,4,mat_idx]),
        # ... and so on ...
        Const(0.0f0), Const(0.0f0), Const(0.0f0), Const(0.0f0), Const(0.0f0), Const(0.0f0), Const(0.0f0), Const(0.0f0),
        Const(Float32(ix)), Const(Float32(iy)), Const(Float32(iz)), Const(shift[1]), Const(shift[2]), Const(shift[3]),
        Const(Float32(src_sh[1])), Const(Float32(src_sh[2])), Const(Float32(src_sh[3]))
    )
    
    # res is a tuple of gradients for Active arguments
    # We would add them to d_mat atomically
end

println("Trying V6 compilation...")
