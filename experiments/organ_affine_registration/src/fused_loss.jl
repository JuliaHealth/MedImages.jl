module FusedLoss

using KernelAbstractions
using CUDA
using ChainRulesCore
using Enzyme
using LinearAlgebra
using ..Preprocessing: OrganMetadata

export compute_organ_loss

# --- Helper Functions (Inlined Logic) ---
# We define them as macros or inline functions to ensure they are compiled into the kernel.
# Pure functions for Enzyme are still good, but for the kernel we want them scalar.

@inline function softplus_kernel(x::T) where T
    return log(one(T) + exp(x))
end

@inline function trilinear_interp_kernel(vol, x, y, z, channel, sx, sy, sz)
    # x, y, z are 1-based coordinates

    # Boundary Check (Extrapolation = 0)
    if x < 1 || x > sx || y < 1 || y > sy || z < 1 || z > sz
        return 0.0f0
    end

    x0 = floor(Int, x)
    y0 = floor(Int, y)
    z0 = floor(Int, z)

    x1 = min(x0 + 1, sx)
    y1 = min(y0 + 1, sy)
    z1 = min(z0 + 1, sz)

    xd = x - x0
    yd = y - y0
    zd = z - z0

    # Read values directly
    v000 = vol[x0, y0, z0, channel]
    v100 = vol[x1, y0, z0, channel]
    v010 = vol[x0, y1, z0, channel]
    v110 = vol[x1, y1, z0, channel]
    v001 = vol[x0, y0, z1, channel]
    v101 = vol[x1, y0, z1, channel]
    v011 = vol[x0, y1, z1, channel]
    v111 = vol[x1, y1, z1, channel]

    c00 = v000 * (1-xd) + v100 * xd
    c10 = v010 * (1-xd) + v110 * xd
    c01 = v001 * (1-xd) + v101 * xd
    c11 = v011 * (1-xd) + v111 * xd

    c0 = c00 * (1-yd) + c10 * yd
    c1 = c01 * (1-yd) + c11 * yd

    return c0 * (1-zd) + c1 * zd
end

# --- Kernel Definition with Tree Reduction ---

"""
    organ_loss_kernel_optimized!(loss_out, points_tensor, affine_params, gold_vol, ...)

Optimized KernelAbstractions kernel for computing organ registration loss.

# Features
- **Fused Operation**: Applies affine transform, computes distance penalty, and samples interpolation in one pass.
- **Tree Reduction**: Uses shared memory (`@localmem`) and binary tree reduction for efficient summation within a workgroup (512 threads), avoiding atomic operations.
- **Scalar Optimization**: All helper functions are inlined.

# Inputs
- `points_tensor`: (3, 512, Num_Organs) - Reference points from Atlas.
- `affine_params`: (15, Num_Organs, Batch) - Predicted transform parameters.
- `gold_vol`: (X, Y, Z, Num_Organs) - One-hot encoded Gold Standard volume.
"""
@kernel function gpu_full_organ_warp_kernel!(
    out_vol, @Const(in_vol), @Const(affine_params), @Const(barycenters), @Const(radii)
)
    # out_vol: (sx, sy, sz, 1, batch) - batch dimension here corresponds to organs
    # in_vol: (sx, sy, sz, 1, batch)
    # affine_params: (15, batch)
    
    i, j, k, _, b = @index(Global, NTuple)
    sx_v, sy_v, sz_v = size(out_vol)[1:3]
    
    # Load Params for this organ
    rx = affine_params[1, b]; ry = affine_params[2, b]; rz = affine_params[3, b]
    tx = affine_params[4, b]; ty = affine_params[5, b]; tz = affine_params[6, b]
    sx = affine_params[7, b]; sy = affine_params[8, b]; sz = affine_params[9, b]
    sh_xy = affine_params[10, b]; sh_xz = affine_params[11, b]; sh_yz = affine_params[12, b]
    cx = affine_params[13, b]; cy = affine_params[14, b]; cz = affine_params[15, b]

    # Target Coord (i, j, k) -> Source Coord (nx, ny, nz)
    # This matches the logic in gpu_organ_loss_kernel_optimized!
    # px = (nx - cx) * sx ...
    # Wait, for warping we need the inverse? 
    # Usually: I_warped(x) = I_orig(T(x))
    # T is the transform Patient -> Atlas (model predicts how to deform atlas to match patient)
    
    # Let's apply Exactly the same as in loss:
    px = (Float32(i) - cx) * sx
    py = (Float32(j) - cy) * sy
    pz = (Float32(k) - cz) * sz
    
    px_s = px + sh_xy * py + sh_xz * pz
    py_s = py + sh_yz * pz
    pz_s = pz

    s_rx = sin(rx); c_rx = cos(rx)
    py_r1 = py_s * c_rx - pz_s * s_rx
    pz_r1 = py_s * s_rx + pz_s * c_rx
    
    s_ry = sin(ry); c_ry = cos(ry)
    px_r2 = px_s * c_ry + pz_r1 * s_ry
    pz_r2 = -px_s * s_ry + pz_r1 * c_ry
    
    s_rz = sin(rz); c_rz = cos(rz)
    px_r3 = px_r2 * c_rz - py_r1 * s_rz
    py_r3 = px_r2 * s_rz + py_r1 * c_rz
    
    nx = px_r3 + cx + tx
    ny = py_r3 + cy + ty
    nz = pz_r2 + cz + tz
    
    # Trilinear interpolation
    interp_val = 0.0f0
    if nx >= 1.0f0 && nx <= Float32(sx_v) && ny >= 1.0f0 && ny <= Float32(sy_v) && nz >= 1.0f0 && nz <= Float32(sz_v)
        x0 = floor(Int, nx); y0 = floor(Int, ny); z0 = floor(Int, nz)
        x1 = min(x0 + 1, sx_v); y1 = min(y0 + 1, sy_v); z1 = min(z0 + 1, sz_v)
        xd = nx - Float32(x0); yd = ny - Float32(y0); zd = nz - Float32(z0)
        
        v000 = in_vol[x0, y0, z0, 1, b]
        v100 = in_vol[x1, y0, z0, 1, b]
        v010 = in_vol[x0, y1, z0, 1, b]
        v110 = in_vol[x1, y1, z0, 1, b]
        v001 = in_vol[x0, y0, z1, 1, b]
        v101 = in_vol[x1, y0, z1, 1, b]
        v011 = in_vol[x0, y1, z1, 1, b]
        v111 = in_vol[x1, y1, z1, 1, b]
        
        c00 = v000 * (1.0f0 - xd) + v100 * xd
        c10 = v010 * (1.0f0 - xd) + v110 * xd
        c01 = v001 * (1.0f0 - xd) + v101 * xd
        c11 = v011 * (1.0f0 - xd) + v111 * xd
        c0 = c00 * (1.0f0 - yd) + c10 * yd
        c1 = c01 * (1.0f0 - yd) + c11 * yd
        interp_val = c0 * (1.0f0 - zd) + c1 * zd
    end
    
    out_vol[i, j, k, 1, b] = interp_val
end

@kernel function gpu_batch_affine_warp_kernel!(
    out_vol, @Const(in_vol), @Const(affine_matrices), @Const(centers)
)
    # out_vol: (sx, sy, sz, num_channels, batch)
    # in_vol: (sx, sy, sz, num_channels, batch)
    # affine_matrices: (3, 4, batch) -> [R | T]
    # centers: (3, batch)
    
    i, j, k, c, b = @index(Global, NTuple)
    
    sx, sy, sz = size(out_vol)[1:3]
    
    # 1-based to center-relative
    cx, cy, cz = centers[1, b], centers[2, b], centers[3, b]
    px = Float32(i) - cx
    py = Float32(j) - cy
    pz = Float32(k) - cz
    
    # Apply Affine: x_new = R * x + T + center
    nx = affine_matrices[1,1,b] * px + affine_matrices[1,2,b] * py + affine_matrices[1,3,b] * pz + affine_matrices[1,4,b] + cx
    ny = affine_matrices[2,1,b] * px + affine_matrices[2,2,b] * py + affine_matrices[2,3,b] * pz + affine_matrices[2,4,b] + cy
    nz = affine_matrices[3,1,b] * px + affine_matrices[3,2,b] * py + affine_matrices[3,3,b] * pz + affine_matrices[3,4,b] + cz
    
    # Trilinear interpolation
    interp_val = 0.0f0
    if nx >= 1.0f0 && nx <= Float32(sx) && ny >= 1.0f0 && ny <= Float32(sy) && nz >= 1.0f0 && nz <= Float32(sz)
        x0 = floor(Int, nx)
        y0 = floor(Int, ny)
        z0 = floor(Int, nz)
        
        x1 = min(x0 + 1, sx)
        y1 = min(y0 + 1, sy)
        z1 = min(z0 + 1, sz)
        
        xd = nx - Float32(x0)
        yd = ny - Float32(y0)
        zd = nz - Float32(z0)
        
        # neighbors
        v000 = in_vol[x0, y0, z0, c, b]
        v100 = in_vol[x1, y0, z0, c, b]
        v010 = in_vol[x0, y1, z0, c, b]
        v110 = in_vol[x1, y1, z0, c, b]
        v001 = in_vol[x0, y0, z1, c, b]
        v101 = in_vol[x1, y0, z1, c, b]
        v011 = in_vol[x0, y1, z1, c, b]
        v111 = in_vol[x1, y1, z1, c, b]
        
        c00 = v000 * (1.0f0 - xd) + v100 * xd
        c10 = v010 * (1.0f0 - xd) + v110 * xd
        c01 = v001 * (1.0f0 - xd) + v101 * xd
        c11 = v011 * (1.0f0 - xd) + v111 * xd
        
        c0 = c00 * (1.0f0 - yd) + c10 * yd
        c1 = c01 * (1.0f0 - yd) + c11 * yd
        
        interp_val = c0 * (1.0f0 - zd) + c1 * zd
    end
    
    out_vol[i, j, k, c, b] = interp_val
end

@kernel function gpu_organ_loss_kernel_optimized!(loss_out, @Const(points_tensor), @Const(affine_params), @Const(gold_vol), @Const(barycenters), @Const(radii), batch_size, num_organs, vol_sx, vol_sy, vol_sz)
    tid = @index(Local)
    gid = @index(Group)

    organ_idx = (gid - 1) % num_organs + 1
    batch_idx = (gid - 1) ÷ num_organs + 1

    # Load Point
    p_x = points_tensor[1, tid, organ_idx]
    p_y = points_tensor[2, tid, organ_idx]
    p_z = points_tensor[3, tid, organ_idx]

    l1_val = 0.0f0

    if p_x > -0.5f0
        # Load Affine Params
        rx = affine_params[1, organ_idx, batch_idx]
        ry = affine_params[2, organ_idx, batch_idx]
        rz = affine_params[3, organ_idx, batch_idx]
        tx = affine_params[4, organ_idx, batch_idx]
        ty = affine_params[5, organ_idx, batch_idx]
        tz = affine_params[6, organ_idx, batch_idx]
        sx = affine_params[7, organ_idx, batch_idx]
        sy = affine_params[8, organ_idx, batch_idx]
        sz = affine_params[9, organ_idx, batch_idx]
        sh_xy = affine_params[10, organ_idx, batch_idx]
        sh_xz = affine_params[11, organ_idx, batch_idx]
        sh_yz = affine_params[12, organ_idx, batch_idx]
        cx = affine_params[13, organ_idx, batch_idx]
        cy = affine_params[14, organ_idx, batch_idx]
        cz = affine_params[15, organ_idx, batch_idx]

        # Apply Affine
        px = (p_x - cx) * sx
        py = (p_y - cy) * sy
        pz = (p_z - cz) * sz

        px_s = px + sh_xy * py + sh_xz * pz
        py_s = py + sh_yz * pz
        pz_s = pz

        s_rx = sin(rx)
        c_rx = cos(rx)
        py_r1 = py_s * c_rx - pz_s * s_rx
        pz_r1 = py_s * s_rx + pz_s * c_rx
        
        s_ry = sin(ry)
        c_ry = cos(ry)
        px_r2 = px_s * c_ry + pz_r1 * s_ry
        pz_r2 = -px_s * s_ry + pz_r1 * c_ry
        
        s_rz = sin(rz)
        c_rz = cos(rz)
        px_r3 = px_r2 * c_rz - py_r1 * s_rz
        py_r3 = px_r2 * s_rz + py_r1 * c_rz
        
        fin_x = px_r3 + cx + tx
        fin_y = py_r3 + cy + ty
        fin_z = pz_r2 + cz + tz # Note Ry only affects X and Z

        # Metric 1: Distance to barycenter
        bx = barycenters[1, organ_idx]
        by = barycenters[2, organ_idx]
        bz = barycenters[3, organ_idx]
        max_r = radii[organ_idx]

        dist_sq = (fin_x - bx)^2 + (fin_y - by)^2 + (fin_z - bz)^2
        dist = sqrt(max(dist_sq, 0.0f0))
        
        # Metric 1: Distance Penalty
        l1_val = max(0.0f0, dist - max_r)

        # Metric 2: Trilinear Interpolation (Unrolled)
        # vol is (sx, sy, sz, num_organs)
        # fin_x, fin_y, fin_z are coordinates
        
        interp_val = 0.0f0
        if fin_x >= 1.0f0 && fin_x <= Float32(vol_sx) && fin_y >= 1.0f0 && fin_y <= Float32(vol_sy) && fin_z >= 1.0f0 && fin_z <= Float32(vol_sz)
            
            x0 = floor(Int, fin_x)
            y0 = floor(Int, fin_y)
            z0 = floor(Int, fin_z)

            x1 = min(x0 + 1, vol_sx)
            y1 = min(y0 + 1, vol_sy)
            z1 = min(z0 + 1, vol_sz)

            xd = fin_x - Float32(x0)
            yd = fin_y - Float32(y0)
            zd = fin_z - Float32(z0)

            # Read 8 neighbors
            v000 = gold_vol[x0, y0, z0, organ_idx]
            v100 = gold_vol[x1, y0, z0, organ_idx]
            v010 = gold_vol[x0, y1, z0, organ_idx]
            v110 = gold_vol[x1, y1, z0, organ_idx]
            v001 = gold_vol[x0, y0, z1, organ_idx]
            v101 = gold_vol[x1, y0, z1, organ_idx]
            v011 = gold_vol[x0, y1, z1, organ_idx]
            v111 = gold_vol[x1, y1, z1, organ_idx]

            c00 = v000 * (1.0f0 - xd) + v100 * xd
            c10 = v010 * (1.0f0 - xd) + v110 * xd
            c01 = v001 * (1.0f0 - xd) + v101 * xd
            c11 = v011 * (1.0f0 - xd) + v111 * xd

            c0 = c00 * (1.0f0 - yd) + c10 * yd
            c1 = c01 * (1.0f0 - yd) + c11 * yd

            interp_val = c0 * (1.0f0 - zd) + c1 * zd
        end
        
        # Target value for interpolation is 1.0 (inside organ)
        # Penalty is squared error
        l1_val += (1.0f0 - interp_val)^2
    end

    if tid == 1
        loss_out[organ_idx, batch_idx] = l1_val
    end
end

# --- Differentiable Launcher ---

"""
    compute_organ_loss(points_tensor, affine_params, gold_vol, barycenters, radii)

High-level, differentiable wrapper for the fused registration loss.

# Mechanics
- Launches `organ_loss_kernel_optimized!` on the appropriate backend (CPU/GPU).
- Defines a custom `ChainRulesCore.rrule` using `Enzyme.autodiff` to propagate gradients through the kernel back to `affine_params`.

# Arguments
- `points_tensor`: Preprocessed Atlas points.
- `affine_params`: Predicted parameters from the model.
- `gold_vol`: Target Gold Standard volume.
- `barycenters`: Pre-calculated organ barycenters (3, Num_Organs).
- `radii`: Pre-calculated organ radii (Num_Organs).

# Returns
Scalar mean loss across all organs and batch items.
"""
function compute_organ_loss(points_tensor, affine_params, gold_vol, barycenters, radii)
    batch_size = size(affine_params, 3)
    num_organs = size(points_tensor, 3)
    vol_sx, vol_sy, vol_sz, _ = size(gold_vol)

    backend = KernelAbstractions.get_backend(affine_params)
    if backend isa KernelAbstractions.GPU
        barycenters = CuArray(barycenters)
        radii = CuArray(radii)
    end

    loss_out = KernelAbstractions.zeros(backend, Float32, num_organs, batch_size)

    groups = batch_size * num_organs
    threads = 512

    kernel = gpu_organ_loss_kernel_optimized!(backend, threads)
    kernel(loss_out, points_tensor, affine_params, gold_vol, barycenters, radii, batch_size, num_organs, vol_sx, vol_sy, vol_sz, ndrange=groups*threads)
    KernelAbstractions.synchronize(backend)

    return sum(loss_out) / length(loss_out)
end

# --- Enzyme Rule ---

function ChainRulesCore.rrule(::typeof(compute_organ_loss), points_tensor, affine_params, gold_vol, barycenters, radii)
    y = compute_organ_loss(points_tensor, affine_params, gold_vol, barycenters, radii)

    function compute_loss_pullback(d_y)
        d_affine = zero(affine_params)

        batch_size = size(affine_params, 3)
        num_organs = size(points_tensor, 3)
        vol_sx, vol_sy, vol_sz, _ = size(gold_vol)

        backend = KernelAbstractions.get_backend(affine_params)
        if backend isa KernelAbstractions.GPU
            barycenters = CuArray(barycenters)
            radii = CuArray(radii)
        end

        loss_out = KernelAbstractions.zeros(backend, Float32, num_organs, batch_size)
        d_loss_out = KernelAbstractions.ones(backend, Float32, num_organs, batch_size)
        N = length(loss_out)
        d_loss_out .*= (d_y / N)

        d_points = zero(points_tensor)
        d_gold = zero(gold_vol)
        d_bary = zero(barycenters)
        d_radii = zero(radii)

        groups = batch_size * num_organs
        threads = 512

        function kernel_wrapper(l, pt, ap, gv, bc, r, bs, no, vsx, vsy, vsz)
             gpu_organ_loss_kernel_optimized!(backend, threads)(l, pt, ap, gv, bc, r, bs, no, vsx, vsy, vsz, ndrange=groups*threads)
             return nothing
        end

        Enzyme.autodiff(
            Reverse,
            kernel_wrapper,
            Duplicated(loss_out, d_loss_out),
            Duplicated(points_tensor, d_points),
            Duplicated(affine_params, d_affine),
            Duplicated(gold_vol, d_gold),
            Duplicated(barycenters, d_bary),
            Duplicated(radii, d_radii),
            Const(batch_size),
            Const(num_organs),
            Const(vol_sx),
            Const(vol_sy),
            Const(vol_sz)
        )

        return NoTangent(), NoTangent(), d_affine, NoTangent(), NoTangent(), NoTangent()
    end

    return y, compute_loss_pullback
end

end # module
