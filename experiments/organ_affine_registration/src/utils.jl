module RegistrationUtils

using ..Basic_transformations
using LinearAlgebra
using KernelAbstractions

export params_to_matrix, apply_random_augmentation

"""
    params_to_matrix(p, center)

Converts a 15-parameter vector into a 4x4 homogenous affine matrix.
Order: Translation * Uncentering * Rotation * Shear * Scale * Centering
"""
function params_to_matrix(p, center)
    # p: rx, ry, rz, tx, ty, tz, sx, sy, sz, sh_xy, sh_xz, sh_yz, cx, cy, cz
    rx, ry, rz = p[1], p[2], p[3]
    tx, ty, tz = p[4], p[5], p[6]
    sx, sy, sz = p[7], p[8], p[9]
    sh_xy, sh_xz, sh_yz = p[10], p[11], p[12]
    # Center is typically from p[13:15], but passed as argument for convenience/overriding
    cx, cy, cz = center 

    # Basic_transformations.create_affine_matrix expects degrees
    rot_deg = (rad2deg(rx), rad2deg(ry), rad2deg(rz))
    
    # Core affine part (Scale * Shear * Rotation)
    M_core = create_affine_matrix(
        translation=(0.0, 0.0, 0.0), # We handle translation separately with uncentering
        rotation=rot_deg,
        scale=(sx, sy, sz),
        shear=(sh_xy, sh_xz, sh_yz)
    )
    
    # Translation and Centering/Uncentering
    M_trans_uncenter = [1.0 0.0 0.0 cx + tx;
                        0.0 1.0 0.0 cy + ty;
                        0.0 0.0 1.0 cz + tz;
                        0.0 0.0 0.0 1.0]
                        
    M_center = [1.0 0.0 0.0 -cx;
                0.0 1.0 0.0 -cy;
                0.0 0.0 1.0 -cz;
                0.0 0.0 0.0 1.0]
                
    return M_trans_uncenter * M_core * M_center
end

"""
    apply_random_augmentation(x, gold, centers, backend)

Applies a random rigid/affine augmentation to the input image and gold standard.
Returns (x_aug, gold_aug, metadata).
"""
function apply_random_augmentation(x, gold, centers, backend)
    # x: (W, H, D, 2, 1)
    # gold: (W, H, D, 4, 1)
    # centers: (3, 1)
    
    # 1. Random Parameters (Rigid-ish)
    rot = (rand(Float32, 3) .- 0.5f0) .* 10.0f0 # +/- 5 degrees
    trans = (rand(Float32, 3) .- 0.5f0) .* 10.0f0 # +/- 5 voxels
    scale = 0.95f0 .+ rand(Float32, 3) .* 0.1f0 # 0.95 to 1.05
    
    # Create Matrix (Center-relative)
    # We use create_affine_matrix from Basic_transformations
    # but we need it as a 3x4 Float32 matrix for the kernel.
    M_full = create_affine_matrix(
        translation=Tuple(trans), 
        rotation=Tuple(rot),
        scale=Tuple(scale)
    )
    
    # We want to map Target -> Source for warping. 
    # M_full maps Source -> Target. So we need the inverse.
    M_inv = inv(M_full)
    
    # 3x4 Float32 on Device
    M_inv_3x4 = Float32.(M_inv[1:3, :])
    M_gpu = KernelAbstractions.allocate(backend, Float32, 3, 4, 1)
    copyto!(M_gpu, reshape(M_inv_3x4, 3, 4, 1))

    # centers must be on device
    centers_gpu = KernelAbstractions.allocate(backend, Float32, size(centers)...)
    copyto!(centers_gpu, centers)
    
    # 2. Warp
    # Use the kernels from FusedLoss
    # We need to import FusedLoss or use the full path.
    # Since we are in RegistrationUtils, and FusedLoss is a parallel module, 
    # we might need to be careful with the namespace.
    # In train.jl everything is included.
    
    # Let's use the kernel from Main.FusedLoss if available, or just re-implement a simple one if not.
    # Actually, the kernels are in fused_loss.jl under FusedLoss module.
    
    x_aug = similar(x)
    gold_aug = similar(gold)
    
    # We need to reach the FusedLoss module. 
    # It's better to move the kernel to a shared location or just implement it here.
    # Copying the warp logic here for self-containedness.
    
    warp_kernel = gpu_batch_affine_warp_kernel_local!(backend, (8, 8, 8))
    
    # Image Aug (2 channels)
    warp_kernel(x_aug, x, M_gpu, centers_gpu, ndrange=size(x_aug))
    # Gold Aug (4 channels)
    warp_kernel(gold_aug, gold, M_gpu, centers_gpu, ndrange=size(gold_aug))
    
    KernelAbstractions.synchronize(backend)
    
    return x_aug, gold_aug, (rot=rot, trans=trans, scale=scale)
end

@kernel function gpu_batch_affine_warp_kernel_local!(
    out_vol, @Const(in_vol), @Const(affine_matrices), @Const(centers)
)
    i, j, k, c, b = @index(Global, NTuple)
    sx, sy, sz = size(out_vol)[1:3]
    cx, cy, cz = centers[1, b], centers[2, b], centers[3, b]
    px = Float32(i) - cx; py = Float32(j) - cy; pz = Float32(k) - cz
    
    nx = affine_matrices[1,1,b] * px + affine_matrices[1,2,b] * py + affine_matrices[1,3,b] * pz + affine_matrices[1,4,b] + cx
    ny = affine_matrices[2,1,b] * px + affine_matrices[2,2,b] * py + affine_matrices[2,3,b] * pz + affine_matrices[2,4,b] + cy
    nz = affine_matrices[3,1,b] * px + affine_matrices[3,2,b] * py + affine_matrices[3,3,b] * pz + affine_matrices[3,4,b] + cz
    
    interp_val = 0.0f0
    if nx >= 1.0f0 && nx <= Float32(sx) && ny >= 1.0f0 && ny <= Float32(sy) && nz >= 1.0f0 && nz <= Float32(sz)
        x0 = floor(Int, nx); y0 = floor(Int, ny); z0 = floor(Int, nz)
        x1 = min(x0 + 1, sx); y1 = min(y0 + 1, sy); z1 = min(z0 + 1, sz)
        xd = nx - Float32(x0); yd = ny - Float32(y0); zd = nz - Float32(z0)
        v000 = in_vol[x0, y0, z0, c, b]; v100 = in_vol[x1, y0, z0, c, b]
        v010 = in_vol[x0, y1, z0, c, b]; v110 = in_vol[x1, y1, z0, c, b]
        v001 = in_vol[x0, y0, z1, c, b]; v101 = in_vol[x1, y0, z1, c, b]
        v011 = in_vol[x0, y1, z1, c, b]; v111 = in_vol[x1, y1, z1, c, b]
        c00 = v000 * (1.0f0-xd) + v100 * xd; c10 = v010 * (1.0f0-xd) + v110 * xd
        c01 = v001 * (1.0f0-xd) + v101 * xd; c11 = v011 * (1.0f0-xd) + v111 * xd
        c0 = c00 * (1.0f0-yd) + c10 * yd; c1 = c01 * (1.0f0-yd) + c11 * yd
        interp_val = c0 * (1.0f0-zd) + c1 * zd
    end
    out_vol[i, j, k, c, b] = interp_val
end

end # module
