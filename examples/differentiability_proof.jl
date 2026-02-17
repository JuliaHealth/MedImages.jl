#=
Differentiability Proof: Learning Inverse 3D Rotations
======================================================

Demonstrates that 3D image rotation is usefully differentiable by training
a CNN+MLP network to predict inverse rotation angles via gradient descent.

Pipeline:
  1. Generate a synthetic 3D line image (imagePrim)
  2. Apply random 3-axis rotations using MedImages.rotate_mi → rotated training data
  3. CNN+MLP processes rotated image → predicts 3 rotation angles
  4. Apply differentiable rotation with predicted angles to the rotated image
  5. L2 loss between result and imagePrim
  6. If loss decreases → rotation is usefully differentiable!

Requirements (add to your Julia environment):
  pkg> add Flux ForwardDiff

Usage:
  julia --project examples/differentiability_proof.jl
=#

using MedImages
using MedImages.MedImage_data_struct
using Flux
using Zygote
using ChainRulesCore
using ForwardDiff
using Statistics
using Random
using LinearAlgebra
using Printf
using Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const IMG_SIZE = 16       # 16×16×16 voxels
const N_TRAIN = 100       # training samples
const N_TEST = 20         # test samples
const ANGLE_RANGE = 30.0  # ±30 degrees
const EPOCHS = 50
const LR = 1e-3

# Fixed datetime to avoid Zygote issues with Dates.now()
const FIXED_DATE = DateTime(2024, 1, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Synthetic Data Generation
# ═══════════════════════════════════════════════════════════════════════════════

"""Create a 3D line image through the center along the z-axis with some thickness."""
function create_line_image(sz::Int)
    img = zeros(Float32, sz, sz, sz)
    c = sz ÷ 2
    for k in 1:sz
        img[c, c, k] = 1.0f0
        # Add thickness for better gradient signal
        if c > 1 && c < sz
            img[c-1, c, k] = 0.3f0
            img[c+1, c, k] = 0.3f0
            img[c, c-1, k] = 0.3f0
            img[c, c+1, k] = 0.3f0
        end
    end
    return img
end

"""Create a MedImage struct from raw voxel data (for use with rotate_mi)."""
function make_medimage(data::Array{Float32,3})
    MedImage(
        voxel_data = data,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        image_type = MedImages.MedImage_data_struct.MRI_type,
        image_subtype = MedImages.MedImage_data_struct.T1_subtype,
        patient_id = "synthetic",
        date_of_saving = FIXED_DATE,
        acquistion_time = FIXED_DATE
    )
end

"""Apply 3-axis rotation using MedImages.rotate_mi (for data generation)."""
function apply_rotations_mi(data::Array{Float32,3}, angles::Vector{Float64})
    mi = make_medimage(copy(data))
    for (axis, angle) in enumerate(angles)
        if abs(angle) > 0.5
            mi = rotate_mi(mi, axis, angle, Linear_en, true)
        end
    end
    out = Float32.(mi.voxel_data)
    sz = size(data)
    if size(out) == sz
        return out
    end
    # Handle size mismatch from rotation by padding/cropping
    result = zeros(Float32, sz)
    rs = min.(sz, size(out))
    result[1:rs[1], 1:rs[2], 1:rs[3]] = out[1:rs[1], 1:rs[2], 1:rs[3]]
    return result
end

"""Generate a dataset of rotated images with their rotation angles."""
function generate_dataset(imagePrim::Array{Float32,3}, n_samples::Int)
    sz = size(imagePrim)
    # Flux Conv3D expects: W × H × D × C × N
    images = zeros(Float32, sz..., 1, n_samples)
    angles_all = zeros(Float32, 3, n_samples)

    for i in 1:n_samples
        θ = Float64.((rand(3) .* 2 .- 1) .* ANGLE_RANGE)
        rotated = apply_rotations_mi(imagePrim, θ)
        images[:,:,:,1,i] = rotated
        angles_all[:,i] = Float32.(θ)
    end

    return images, angles_all
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Differentiable 3D Rotation
#
# Pure Julia implementation using Euler rotation matrices and trilinear
# interpolation. Gradients w.r.t. rotation angles are computed via
# ForwardDiff through a custom ChainRulesCore rrule, enabling seamless
# integration with Zygote's reverse-mode AD for the neural network.
#
# NOTE: MedImages' interpolate_pure already computes coordinate gradients
# via Enzyme. To make rotate_mi itself angle-differentiable, one would
# only need to rewrite the rotation matrix construction in pure Julia
# (removing @non_differentiable annotations). This standalone version
# serves as a clean proof-of-concept.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    euler_rotation_matrix(angles_deg)

Build a 3×3 rotation matrix from Euler angles (degrees): Rz(az) * Ry(ay) * Rx(ax).
Pure Julia (sin/cos only) — compatible with ForwardDiff Dual numbers.
"""
function euler_rotation_matrix(angles_deg)
    d2r = π / 180
    ax = angles_deg[1] * d2r
    ay = angles_deg[2] * d2r
    az = angles_deg[3] * d2r

    c1, s1 = cos(ax), sin(ax)
    c2, s2 = cos(ay), sin(ay)
    c3, s3 = cos(az), sin(az)

    # Rz * Ry * Rx  (column-major layout for reshape)
    reshape([
        c2*c3,           c2*s3,           -s2,
        s1*s2*c3-c1*s3,  s1*s2*s3+c1*c3,  s1*c2,
        c1*s2*c3+s1*s3,  c1*s2*s3-s1*c3,  c1*c2
    ], 3, 3)
end

"""
    trilinear_sample(img, x, y, z)

Sample a 3D array at continuous coordinates (x,y,z) using trilinear interpolation.
Works with ForwardDiff Dual numbers: gradients flow through the interpolation
weights (fractional parts) while integer floor indices are non-differentiable.
"""
function trilinear_sample(img, x, y, z)
    sx, sy, sz = size(img)

    x = clamp(x, 1.0, Float64(sx))
    y = clamp(y, 1.0, Float64(sy))
    z = clamp(z, 1.0, Float64(sz))

    x0 = clamp(floor(Int, x), 1, sx)
    y0 = clamp(floor(Int, y), 1, sy)
    z0 = clamp(floor(Int, z), 1, sz)
    x1 = min(x0 + 1, sx)
    y1 = min(y0 + 1, sy)
    z1 = min(z0 + 1, sz)

    xd = x - x0
    yd = y - y0
    zd = z - z0

    # Gather 8 corners
    c000 = img[x0,y0,z0]; c100 = img[x1,y0,z0]
    c010 = img[x0,y1,z0]; c110 = img[x1,y1,z0]
    c001 = img[x0,y0,z1]; c101 = img[x1,y0,z1]
    c011 = img[x0,y1,z1]; c111 = img[x1,y1,z1]

    # Trilinear blend
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    c0  = c00  * (1 - yd) + c10  * yd
    c1  = c01  * (1 - yd) + c11  * yd
    return c0 * (1 - zd) + c1 * zd
end

"""Pre-compute centered grid coordinates for rotation (constant, not differentiated)."""
function make_grid(sx, sy, sz)
    cx = Float32((sx + 1) / 2)
    cy = Float32((sy + 1) / 2)
    cz = Float32((sz + 1) / 2)
    N = sx * sy * sz
    g = Matrix{Float32}(undef, 3, N)
    idx = 1
    for k in 1:sz, j in 1:sy, i in 1:sx
        g[1, idx] = i - cx
        g[2, idx] = j - cy
        g[3, idx] = k - cz
        idx += 1
    end
    return g
end

"""
Core rotation implementation. Rotates a 3D image by Euler angles using trilinear
interpolation. No AD-specific constructs — works with plain values and ForwardDiff Duals.
"""
function _rotate_impl(img::Array{Float32,3}, angles_deg, grid::Matrix{Float32})
    R_inv = euler_rotation_matrix(angles_deg)'
    src = R_inv * grid

    sx, sy, sz = size(img)
    cx = Float64((sx + 1) / 2)
    cy = Float64((sy + 1) / 2)
    cz = Float64((sz + 1) / 2)
    src = src .+ [cx, cy, cz]

    N = size(src, 2)
    output = Vector{eltype(src)}(undef, N)
    for n in 1:N
        output[n] = trilinear_sample(img, src[1,n], src[2,n], src[3,n])
    end

    return reshape(output, sx, sy, sz)
end

"""
    diff_rotate_3d(img, angles_deg, grid)

Differentiable 3D rotation. Forward pass uses `_rotate_impl`; backward pass
computes exact angle gradients via ForwardDiff (forward-mode AD through the
rotation matrix and trilinear interpolation).
"""
function diff_rotate_3d(img::Array{Float32,3}, angles_deg, grid::Matrix{Float32})
    return _rotate_impl(img, angles_deg, grid)
end

# Custom reverse-mode rule: angle gradients via ForwardDiff Jacobian
function ChainRulesCore.rrule(::typeof(diff_rotate_3d), img, angles_deg, grid)
    output = _rotate_impl(img, Float32.(angles_deg), grid)

    function diff_rotate_pullback(Δ)
        # ForwardDiff computes the exact Jacobian ∂output/∂angles (forward-mode).
        # Only 3 forward passes (one per angle), each over N=sx*sy*sz voxels.
        angles_f64 = Float64.(angles_deg)
        J = ForwardDiff.jacobian(
            a -> vec(_rotate_impl(img, a, grid)),
            angles_f64
        )
        # Vector-Jacobian product: d_angles = J' * Δ_flat
        d_angles = Float32.(J' * vec(Float64.(Δ)))
        return NoTangent(), NoTangent(), d_angles, NoTangent()
    end

    return output, diff_rotate_pullback
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Neural Network Model
# ═══════════════════════════════════════════════════════════════════════════════

"""Build a simple 3D CNN + MLP that predicts 3 rotation angles from a 3D image."""
function build_model()
    Chain(
        # 3D CNN feature extraction
        Conv((3,3,3), 1 => 8, relu; pad=1),
        MaxPool((2,2,2)),
        Conv((3,3,3), 8 => 16, relu; pad=1),
        MaxPool((2,2,2)),
        Conv((3,3,3), 16 => 32, relu; pad=1),
        GlobalMeanPool(),
        Flux.flatten,
        # MLP head: predict 3 rotation angles
        Dense(32 => 64, relu),
        Dense(64 => 3)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

"""Compute L2 loss for a single sample: rotate with predicted angles, compare to target."""
function compute_loss(model, x_5d, img_3d, imagePrim, grid)
    pred_angles = vec(model(x_5d))
    reconstructed = diff_rotate_3d(img_3d, pred_angles, grid)
    return sum((reconstructed .- imagePrim) .^ 2) / length(imagePrim)
end

function main()
    Random.seed!(42)

    println("=" ^ 65)
    println("  Differentiability Proof: Learning Inverse 3D Rotations")
    println("  MedImages.jl")
    println("=" ^ 65)

    # ─── Generate Data ───────────────────────────────────────────────────────
    println("\n[1/4] Generating synthetic data...")
    imagePrim = create_line_image(IMG_SIZE)
    println("  Image size: $(size(imagePrim))")
    println("  Angle range: +/-$(ANGLE_RANGE) degrees")

    print("  Generating $(N_TRAIN) training samples with rotate_mi... ")
    train_imgs, train_angles = generate_dataset(imagePrim, N_TRAIN)
    println("done")

    print("  Generating $(N_TEST) test samples with rotate_mi... ")
    test_imgs, test_angles = generate_dataset(imagePrim, N_TEST)
    println("done")

    # Baseline: L2 loss without any correction
    baseline_loss = mean([
        sum((train_imgs[:,:,:,1,i] .- imagePrim).^2) / length(imagePrim)
        for i in 1:N_TRAIN
    ])
    @printf("  Baseline L2 loss (no correction): %.6f\n", baseline_loss)

    # ─── Build Model ─────────────────────────────────────────────────────────
    println("\n[2/4] Building CNN + MLP model...")
    model = build_model()
    opt_state = Flux.setup(Adam(LR), model)
    # Count parameters
    n_params = sum(length(p) for p in Flux.trainables(model))
    println("  Total parameters: $(n_params)")

    # Pre-compute rotation grid (constant across all samples)
    grid = make_grid(IMG_SIZE, IMG_SIZE, IMG_SIZE)

    # ─── Train ───────────────────────────────────────────────────────────────
    println("\n[3/4] Training ($(EPOCHS) epochs, $(N_TRAIN) samples)...")
    println("  Gradients flow: L2 loss -> trilinear interp -> rotation matrix -> CNN/MLP")
    println("-" ^ 65)
    @printf("  %-8s  %-18s  %-18s\n", "Epoch", "Avg Train Loss", "Avg Test Loss")
    println("-" ^ 65)

    train_history = Float32[]
    test_history = Float32[]

    for epoch in 1:EPOCHS
        perm = randperm(N_TRAIN)
        epoch_loss = 0.0f0

        for i in perm
            x_5d = train_imgs[:,:,:,:,i:i]
            img_3d = train_imgs[:,:,:,1,i]

            loss_val, grads = Flux.withgradient(model) do m
                compute_loss(m, x_5d, img_3d, imagePrim, grid)
            end

            Flux.update!(opt_state, model, grads[1])
            epoch_loss += loss_val
        end

        avg_train = epoch_loss / N_TRAIN
        push!(train_history, avg_train)

        # Evaluate on test set periodically
        if epoch == 1 || epoch % 10 == 0 || epoch == EPOCHS
            test_loss = 0.0f0
            for i in 1:N_TEST
                x_5d = test_imgs[:,:,:,:,i:i]
                img_3d = test_imgs[:,:,:,1,i]
                test_loss += compute_loss(model, x_5d, img_3d, imagePrim, grid)
            end
            avg_test = test_loss / N_TEST
            push!(test_history, avg_test)
            @printf("  %-8d  %-18.6f  %-18.6f\n", epoch, avg_train, avg_test)
        end
    end

    println("-" ^ 65)

    # ─── Results ─────────────────────────────────────────────────────────────
    println("\n[4/4] Results")
    println("=" ^ 65)

    initial_test = test_history[1]
    final_test = test_history[end]

    @printf("  Baseline loss (no correction):     %.6f\n", baseline_loss)
    @printf("  Initial loss  (untrained model):   %.6f\n", initial_test)
    @printf("  Final loss    (trained model):     %.6f\n", final_test)
    @printf("  Improvement over baseline:         %.1f%%\n", (1 - final_test/baseline_loss) * 100)
    @printf("  Improvement over initial:          %.1f%%\n", (1 - final_test/initial_test) * 100)
    println()

    if final_test < baseline_loss * 0.7
        println("  SUCCESS: Loss decreased significantly!")
        println("  The rotation function is usefully differentiable.")
        println("  Gradients flow through: L2 loss -> interpolation -> rotation angles -> CNN/MLP")
    elseif final_test < baseline_loss * 0.95
        println("  PARTIAL: Loss decreased. More training may improve results.")
    else
        println("  Loss did not decrease enough. Try more epochs or smaller angle range.")
    end

    println()
    println("  Technical notes:")
    println("  - Data generated with MedImages.rotate_mi (3-axis sequential rotation)")
    println("  - Training rotation: Euler angles Rz*Ry*Rx with trilinear interpolation")
    println("  - Angle gradients: exact via ForwardDiff (forward-mode AD)")
    println("  - Network gradients: Zygote (reverse-mode AD)")
    println("  - MedImages' interpolate_pure already supports coordinate gradients")
    println("    via Enzyme, so making rotate_mi angle-differentiable only requires")
    println("    rewriting the rotation matrix construction in pure Julia.")
    println("=" ^ 65)

    return model, train_history, test_history
end

# Run the proof
main()
