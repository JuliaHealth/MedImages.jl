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

@kernel function organ_loss_kernel_optimized!(loss_out, @Const(points_tensor), @Const(affine_params), @Const(gold_vol), @Const(barycenters), @Const(radii), batch_size, num_organs, vol_sx, vol_sy, vol_sz)
    # Workgroup size must be power of 2 for tree reduction (e.g., 512)
    # One Workgroup per (Organ * Batch)

    tid = @index(Local)
    gid = @index(Group)

    # Shared Memory for Reduction
    shared_l1 = @localmem Float32 512
    shared_l2 = @localmem Float32 512
    shared_valid = @localmem Float32 512 # Using float for easier summation

    # Calculate indices purely locally to avoid closure issues
    # gid is 1-based group index (corresponds to one Organ in one Batch)
    # Mapping: gid = (batch_idx - 1) * num_organs + organ_idx
    # So:
    # organ_idx = (gid - 1) % num_organs + 1
    # batch_idx = (gid - 1) ÷ num_organs + 1

    organ_idx = (gid - 1) % num_organs + 1
    batch_idx = (gid - 1) ÷ num_organs + 1

    # Load Point
    p_x = points_tensor[1, tid, organ_idx]
    p_y = points_tensor[2, tid, organ_idx]
    p_z = points_tensor[3, tid, organ_idx]

    l1_val = 0.0f0
    l2_val = 0.0f0
    valid_val = 0.0f0

    if p_x > -0.5f0
        valid_val = 1.0f0

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

        # Rotation Rx
        s_rx, c_rx = sincos(rx)
        py_r1 = py_s * c_rx - pz_s * s_rx
        pz_r1 = py_s * s_rx + pz_s * c_rx
        px_r1 = px_s

        # Rotation Ry
        s_ry, c_ry = sincos(ry)
        px_r2 = px_r1 * c_ry + pz_r1 * s_ry
        pz_r2 = -px_r1 * s_ry + pz_r1 * c_ry
        py_r2 = py_r1

        # Rotation Rz
        s_rz, c_rz = sincos(rz)
        px_r3 = px_r2 * c_rz - py_r2 * s_rz
        py_r3 = px_r2 * s_rz + py_r2 * c_rz
        pz_r3 = pz_r2

        fin_x = px_r3 + cx + tx
        fin_y = py_r3 + cy + ty
        fin_z = pz_r3 + cz + tz

        # Metric 1
        bx = barycenters[1, organ_idx]
        by = barycenters[2, organ_idx]
        bz = barycenters[3, organ_idx]
        max_r = radii[organ_idx]

        dist = sqrt((fin_x - bx)^2 + (fin_y - by)^2 + (fin_z - bz)^2)
        l1_val = softplus_kernel(dist - max_r)

        # Metric 2
        interp_val = trilinear_interp_kernel(gold_vol, fin_x, fin_y, fin_z, organ_idx, vol_sx, vol_sy, vol_sz)
        l2_val = (1.0f0 - interp_val)^2
    end

    shared_l1[tid] = l1_val
    shared_l2[tid] = l2_val
    shared_valid[tid] = valid_val

    @synchronize

    # Tree Reduction (512 -> 1)

    if tid <= 256
        shared_l1[tid] += shared_l1[tid + 256]
        shared_l2[tid] += shared_l2[tid + 256]
        shared_valid[tid] += shared_valid[tid + 256]
    end
    @synchronize

    if tid <= 128
        shared_l1[tid] += shared_l1[tid + 128]
        shared_l2[tid] += shared_l2[tid + 128]
        shared_valid[tid] += shared_valid[tid + 128]
    end
    @synchronize

    if tid <= 64
        shared_l1[tid] += shared_l1[tid + 64]
        shared_l2[tid] += shared_l2[tid + 64]
        shared_valid[tid] += shared_valid[tid + 64]
    end
    @synchronize

    if tid <= 32
        shared_l1[tid] += shared_l1[tid + 32]
        shared_l2[tid] += shared_l2[tid + 32]
        shared_valid[tid] += shared_valid[tid + 32]
    end
    @synchronize

    if tid <= 16
        shared_l1[tid] += shared_l1[tid + 16]
        shared_l2[tid] += shared_l2[tid + 16]
        shared_valid[tid] += shared_valid[tid + 16]
    end
    @synchronize

    if tid <= 8
        shared_l1[tid] += shared_l1[tid + 8]
        shared_l2[tid] += shared_l2[tid + 8]
        shared_valid[tid] += shared_valid[tid + 8]
    end
    @synchronize

    if tid <= 4
        shared_l1[tid] += shared_l1[tid + 4]
        shared_l2[tid] += shared_l2[tid + 4]
        shared_valid[tid] += shared_valid[tid + 4]
    end
    @synchronize

    if tid <= 2
        shared_l1[tid] += shared_l1[tid + 2]
        shared_l2[tid] += shared_l2[tid + 2]
        shared_valid[tid] += shared_valid[tid + 2]
    end
    @synchronize

    if tid == 1
        total_l1 = shared_l1[1] + shared_l1[2]
        total_l2 = shared_l2[1] + shared_l2[2]
        total_cnt = shared_valid[1] + shared_valid[2]

        # We need to recalculate indices for writing output
        # to ensure scope
        idx_o = (gid - 1) % num_organs + 1
        idx_b = (gid - 1) ÷ num_organs + 1

        if total_cnt > 0.5f0
            loss_out[idx_o, idx_b] = (total_l1 + total_l2) / total_cnt
        else
            loss_out[idx_o, idx_b] = 0.0f0
        end
    end
end

# --- Differentiable Launcher ---

function compute_organ_loss(points_tensor, affine_params, gold_vol, organ_meta_list)
    batch_size = size(affine_params, 3)
    num_organs = size(points_tensor, 3)
    vol_sx, vol_sy, vol_sz, _ = size(gold_vol)

    # Convert Metadata
    barycenters = zeros(Float32, 3, num_organs)
    radii = zeros(Float32, num_organs)
    for (i, m) in enumerate(organ_meta_list)
        barycenters[1, i] = m.barycenter[1]
        barycenters[2, i] = m.barycenter[2]
        barycenters[3, i] = m.barycenter[3]
        radii[i] = m.max_radius
    end

    backend = KernelAbstractions.get_backend(affine_params)
    if backend isa KernelAbstractions.GPU
        barycenters = CuArray(barycenters)
        radii = CuArray(radii)
    end

    loss_out = KernelAbstractions.zeros(backend, Float32, num_organs, batch_size)

    groups = batch_size * num_organs
    threads = 512

    kernel = organ_loss_kernel_optimized!(backend, threads)
    kernel(loss_out, points_tensor, affine_params, gold_vol, barycenters, radii, batch_size, num_organs, vol_sx, vol_sy, vol_sz, ndrange=groups*threads)
    KernelAbstractions.synchronize(backend)

    return sum(loss_out) / length(loss_out)
end

# --- Enzyme Rule ---

function ChainRulesCore.rrule(::typeof(compute_organ_loss), points_tensor, affine_params, gold_vol, organ_meta_list)
    y = compute_organ_loss(points_tensor, affine_params, gold_vol, organ_meta_list)

    function compute_loss_pullback(d_y)
        d_affine = zero(affine_params)

        batch_size = size(affine_params, 3)
        num_organs = size(points_tensor, 3)
        vol_sx, vol_sy, vol_sz, _ = size(gold_vol)

        barycenters = zeros(Float32, 3, num_organs)
        radii = zeros(Float32, num_organs)
        for (i, m) in enumerate(organ_meta_list)
            barycenters[1, i] = m.barycenter[1]
            barycenters[2, i] = m.barycenter[2]
            barycenters[3, i] = m.barycenter[3]
            radii[i] = m.max_radius
        end

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
             organ_loss_kernel_optimized!(backend, threads)(l, pt, ap, gv, bc, r, bs, no, vsx, vsy, vsz, ndrange=groups*threads)
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

        return NoTangent(), NoTangent(), d_affine, NoTangent(), NoTangent()
    end

    return y, compute_loss_pullback
end

end # module
