module FusedLoss

using KernelAbstractions
using CUDA
using ChainRulesCore
using Enzyme
using LinearAlgebra
using ..Preprocessing: OrganMetadata

export compute_organ_loss

# --- Helper Functions (Pure for Enzyme) ---

# Softplus: log(1 + exp(x))
function softplus_pure(x::T) where T
    return log(one(T) + exp(x))
end

# Trilinear Interpolation (Single Point)
function trilinear_interp_pure(vol::AbstractArray{T, 4}, x::T, y::T, z::T, channel::Int) where T
    # x, y, z are 1-based coordinates
    sx, sy, sz, _ = size(vol)

    # Boundary Check (Extrapolation = 0)
    if x < 1 || x > sx || y < 1 || y > sy || z < 1 || z > sz
        return zero(T)
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

    # Read values
    c00 = vol[x0, y0, z0, channel] * (1-xd) + vol[x1, y0, z0, channel] * xd
    c10 = vol[x0, y1, z0, channel] * (1-xd) + vol[x1, y1, z0, channel] * xd
    c01 = vol[x0, y0, z1, channel] * (1-xd) + vol[x1, y0, z1, channel] * xd
    c11 = vol[x0, y1, z1, channel] * (1-xd) + vol[x1, y1, z1, channel] * xd

    c0 = c00 * (1-yd) + c10 * yd
    c1 = c01 * (1-yd) + c11 * yd

    return c0 * (1-zd) + c1 * zd
end

# --- Kernel Definition ---

@kernel function organ_loss_kernel!(loss_out, @Const(points_tensor), @Const(affine_params), @Const(gold_vol), @Const(barycenters), @Const(radii), batch_size, num_organs)
    # Thread Layout:
    # 1 Workgroup per (Organ * Batch)
    # Workgroup size = 512 (Max Points)

    tid = @index(Local)
    gid = @index(Group)

    # Use Shared Memory for Indices to avoid capture issues
    shared_organ_idx = @localmem Int32 1
    shared_batch_idx = @localmem Int32 1

    # Local accumulation
    local_loss1 = @localmem Float32 1
    local_loss2 = @localmem Float32 1
    local_count = @localmem Int32 1

    if tid == 1
        # Map Group ID to (Batch, Organ)
        # gid ranges 1 to (Batch * Num_Organs)

        # Calculation must be explicit
        o_idx = (gid - 1) % num_organs + 1
        b_idx = (gid - 1) ÷ num_organs + 1

        shared_organ_idx[1] = Int32(o_idx)
        shared_batch_idx[1] = Int32(b_idx)

        local_loss1[1] = 0.0f0
        local_loss2[1] = 0.0f0
        local_count[1] = 0
    end
    @synchronize

    # Read from shared memory
    organ_idx = Int(shared_organ_idx[1])
    batch_idx = Int(shared_batch_idx[1])

    # Load Point
    # points_tensor: (3, 512, Num_Organs)
    p_x = points_tensor[1, tid, organ_idx]
    p_y = points_tensor[2, tid, organ_idx]
    p_z = points_tensor[3, tid, organ_idx]

    l1 = 0.0f0
    l2 = 0.0f0
    valid = 0

    if p_x > -0.5f0 # Not -1 (padded)
        valid = 1

        # Load Affine Params for this Organ & Batch
        # params: (15, Num_Organs, Batch)
        # 1-3: Rot, 4-6: Trans, 7-9: Scale, 10-12: Shear, 13-15: Center

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

        # Construct Matrix (on the fly to save memory/registers)
        # Center Shift
        px = p_x - cx
        py = p_y - cy
        pz = p_z - cz

        # Scale
        px *= sx
        py *= sy
        pz *= sz

        # Shear
        px_s = px + sh_xy * py + sh_xz * pz
        py_s = py + sh_yz * pz
        pz_s = pz

        # Rotation (XYZ order)
        # Rx
        sin_rx, cos_rx = sincos(rx)
        py_r1 = py_s * cos_rx - pz_s * sin_rx
        pz_r1 = py_s * sin_rx + pz_s * cos_rx
        px_r1 = px_s

        # Ry
        sin_ry, cos_ry = sincos(ry)
        px_r2 = px_r1 * cos_ry + pz_r1 * sin_ry
        pz_r2 = -px_r1 * sin_ry + pz_r1 * cos_ry
        py_r2 = py_r1

        # Rz
        sin_rz, cos_rz = sincos(rz)
        px_r3 = px_r2 * cos_rz - py_r2 * sin_rz
        py_r3 = px_r2 * sin_rz + py_r2 * cos_rz
        pz_r3 = pz_r2

        # Translation + Back Shift
        fin_x = px_r3 + cx + tx
        fin_y = py_r3 + cy + ty
        fin_z = pz_r3 + cz + tz

        # Metric 1: Distance Penalty
        # Barycenter for organ_idx
        bx = barycenters[1, organ_idx]
        by = barycenters[2, organ_idx]
        bz = barycenters[3, organ_idx]
        max_r = radii[organ_idx]

        dist = sqrt((fin_x - bx)^2 + (fin_y - by)^2 + (fin_z - bz)^2)
        diff = dist - max_r
        # Softplus penalty
        l1 = log(1.0f0 + exp(diff)) # Softplus

        # Metric 2: Interpolation Loss
        # Look in gold_vol (4D) at channel organ_idx
        val = trilinear_interp_pure(gold_vol, fin_x, fin_y, fin_z, organ_idx)
        l2 = (1.0f0 - val)^2
    end

    # Atomic Reduction (Simple but effective for 512 threads)
    # Atomix.jl is preferred, but for now we use CUDA/KA atomics if available
    # Or simple loop in shared mem?
    # KernelAbstractions atomic operations

    KernelAbstractions.@atomic local_loss1[1] += l1
    KernelAbstractions.@atomic local_loss2[1] += l2
    KernelAbstractions.@atomic local_count[1] += valid

    @synchronize

    if tid == 1
        cnt = local_count[1]
        if cnt > 0
            # Write average loss for this organ
            # loss_out: (Num_Organs, Batch)
            avg_l1 = local_loss1[1] / Float32(cnt)
            avg_l2 = local_loss2[1] / Float32(cnt)
            loss_out[organ_idx, batch_idx] = avg_l1 + avg_l2
        else
            loss_out[organ_idx, batch_idx] = 0.0f0
        end
    end
end

# --- Differentiable Launcher ---

function compute_organ_loss(points_tensor, affine_params, gold_vol, organ_meta_list)
    batch_size = size(affine_params, 3)
    num_organs = size(points_tensor, 3)

    # Convert Metadata to Arrays for Kernel
    barycenters = zeros(Float32, 3, num_organs)
    radii = zeros(Float32, num_organs)

    for (i, m) in enumerate(organ_meta_list)
        barycenters[1, i] = m.barycenter[1]
        barycenters[2, i] = m.barycenter[2]
        barycenters[3, i] = m.barycenter[3]
        radii[i] = m.max_radius
    end

    backend = KernelAbstractions.get_backend(affine_params)

    # Move meta to backend
    # Assume affine_params determines backend
    if backend isa KernelAbstractions.GPU
        barycenters = CuArray(barycenters)
        radii = CuArray(radii)
        # points and gold_vol should already be on device
    end

    loss_out = KernelAbstractions.zeros(backend, Float32, num_organs, batch_size)

    # Launch
    # Groups: Batch * Num_Organs
    # Threads: 512
    groups = batch_size * num_organs
    threads = 512

    kernel = organ_loss_kernel!(backend, threads)
    kernel(loss_out, points_tensor, affine_params, gold_vol, barycenters, radii, batch_size, num_organs, ndrange=groups*threads)
    KernelAbstractions.synchronize(backend)

    # Return mean loss
    return sum(loss_out) / length(loss_out)
end

# --- Enzyme Rule ---

function ChainRulesCore.rrule(::typeof(compute_organ_loss), points_tensor, affine_params, gold_vol, organ_meta_list)
    # Forward Pass
    y = compute_organ_loss(points_tensor, affine_params, gold_vol, organ_meta_list)

    function compute_loss_pullback(d_y)
        # d_y is the scalar gradient of the mean loss
        # We need gradients w.r.t affine_params

        d_affine = zero(affine_params)

        # Prepare inputs exactly as in forward
        batch_size = size(affine_params, 3)
        num_organs = size(points_tensor, 3)

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
        # Scale d_loss_out by d_y / N (since y = sum(loss_out)/N)
        N = length(loss_out)
        d_loss_out .*= (d_y / N)

        # Shadow copies for Enzyme
        d_points = zero(points_tensor)
        d_gold = zero(gold_vol) # We don't differentiate gold, but Enzyme needs shadow
        d_bary = zero(barycenters)
        d_radii = zero(radii)

        # Launch Autodiff
        groups = batch_size * num_organs
        threads = 512

        # Using Enzyme.autodiff on the kernel
        # We need to wrap the kernel call
        function kernel_wrapper(l, pl, pt, ap, gv, bc, r, bs, no)
             organ_loss_kernel!(backend, threads)(l, pt, ap, gv, bc, r, bs, no, ndrange=groups*threads)
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
            Const(num_organs)
        )

        return NoTangent(), NoTangent(), d_affine, NoTangent(), NoTangent()
    end

    return y, compute_loss_pullback
end

end # module
