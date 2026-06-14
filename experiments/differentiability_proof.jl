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
using Enzyme
using Statistics
using Random
using LinearAlgebra
using Printf
using Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

const IMG_SIZE = 32       # 32×32×32 voxels
const N_TRAIN = 100       # training samples
const N_TEST = 20         # test samples
const ANGLE_RANGE = 30.0  # ±30 degrees
const EPOCHS = 100
const LR = 5e-4

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
# 1.5 Extract Endpoints Helper
# ═══════════════════════════════════════════════════════════════════════════════
function extract_endpoints(img::Array{Float32, 3}, threshold=0.1f0)
    coords = findall(img .> threshold)
    if isempty(coords)
        return [16.0, 16.0, 16.0], [16.0, 16.0, 16.0]
    end
    N = length(coords)
    X = zeros(Float64, N, 3)
    for i in 1:N
        X[i, 1] = coords[i][1] - 1
        X[i, 2] = coords[i][2] - 1
        X[i, 3] = coords[i][3] - 1
    end
    
    μ = mean(X, dims=1)
    X_c = X .- μ
    Cov = (X_c' * X_c) ./ (N - 1)
    
    F = eigen(Cov)
    idx = argmax(real.(F.values))
    principal_axis = real.(F.vectors[:, idx])
    
    projections = X_c * principal_axis
    min_idx = argmin(projections[:, 1])
    max_idx = argmax(projections[:, 1])
    
    p1 = X[min_idx, :]
    p2 = X[max_idx, :]
    
    return p1, p2
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Differentiable 3D Rotation
# ═══════════════════════════════════════════════════════════════════════════════

using MedImages.Utils: interpolate_fused_affine
using MedImages.MedImage_data_struct: Linear_en

function euler_rotation_matrix(angles_deg)
    d2r = π / 180
    ax = angles_deg[1] * d2r
    ay = angles_deg[2] * d2r
    az = angles_deg[3] * d2r

    c1, s1 = cos(ax), sin(ax)
    c2, s2 = cos(ay), sin(ay)
    c3, s3 = cos(az), sin(az)

    # Rz * Ry * Rx
    return [
        c2*c3           c2*s3           -s2;
        s1*s2*c3-c1*s3  s1*s2*s3+c1*c3  s1*c2;
        c1*s2*c3+s1*s3  c1*s2*s3-s1*c3  c1*c2
    ]
end

function diff_rotate_3d(img::Array{Float32,3}, angles_deg)
    R_inv = Float32.(euler_rotation_matrix(angles_deg)')
    
    M_inv = [R_inv[1,1] R_inv[1,2] R_inv[1,3] 0.0f0;
             R_inv[2,1] R_inv[2,2] R_inv[2,3] 0.0f0;
             R_inv[3,1] R_inv[3,2] R_inv[3,3] 0.0f0;
             0.0f0      0.0f0      0.0f0      1.0f0]
    
    M_inv_3d = reshape(M_inv, 4, 4, 1)
    output_size = size(img)
    reconstructed_flat = interpolate_fused_affine(img, M_inv_3d, output_size, Linear_en, false, 0.0f0, nothing)
    
    return reshape(reconstructed_flat, output_size...)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Neural Network Model
# ═══════════════════════════════════════════════════════════════════════════════

function build_model()
    Chain(
        Conv((3,3,3), 1 => 16, relu; pad=1),
        MaxPool((2,2,2)),
        Conv((3,3,3), 16 => 32, relu; pad=1),
        Conv((3,3,3), 32 => 32, relu; pad=1),
        MaxPool((2,2,2)),
        Conv((3,3,3), 32 => 64, relu; pad=1),
        Conv((3,3,3), 64 => 64, relu; pad=1),
        GlobalMeanPool(),
        Flux.flatten,
        Dense(64 => 128, relu),
        Dense(128 => 128, relu),
        Dense(128 => 3)
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

function compute_metrics(model, x_5d, img_3d, imagePrim)
    pred_angles = vec(model(x_5d))
    reconstructed = diff_rotate_3d(img_3d, pred_angles)
    mse = sum((reconstructed .- imagePrim) .^ 2) / length(imagePrim)
    mae = sum(abs.(reconstructed .- imagePrim)) / length(imagePrim)
    return mse, mae
end

const OUTPUT_ARTIFACT_DIR = "data/validation"
mkpath(OUTPUT_ARTIFACT_DIR)

function main()
    Random.seed!(42)

    println("=" ^ 65)
    println("  Differentiability Proof: Learning Inverse 3D Rotations")
    println("=" ^ 65)

    println("\n[1/4] Generating synthetic data...")
    imagePrim = create_line_image(IMG_SIZE)
    
    # Save the Gold Standard for plotting
    MedImages.Load_and_save.create_nii_from_medimage(make_medimage(imagePrim), joinpath(OUTPUT_ARTIFACT_DIR, "gold_standard.nii.gz"))

    train_imgs, train_angles = generate_dataset(imagePrim, N_TRAIN)
    test_imgs, test_angles = generate_dataset(imagePrim, N_TEST)
    
    # Save the uncorrected (initial) rotated image for test sample 1
    MedImages.Load_and_save.create_nii_from_medimage(make_medimage(test_imgs[:,:,:,1,1]), joinpath(OUTPUT_ARTIFACT_DIR, "uncorrected.nii.gz"))

    baseline_mse = mean([
        sum((train_imgs[:,:,:,1,i] .- imagePrim).^2) / length(imagePrim)
        for i in 1:N_TRAIN
    ])
    @printf("  Baseline L2 loss (no correction): %.6f\n", baseline_mse)

    println("\n[2/4] Building CNN + MLP model...")
    model = build_model()
    opt_state = Flux.setup(Adam(LR), model)

    println("\n[3/4] Training ($(EPOCHS) epochs)...")
    
    # Track metrics
    endpoints_history = []
    history = []
    
    for epoch in 1:EPOCHS
        perm = randperm(N_TRAIN)
        epoch_mse = 0.0f0
        epoch_mae = 0.0f0
        for i in perm
            x_5d = train_imgs[:,:,:,:,i:i]
            img_3d = train_imgs[:,:,:,1,i]
            (loss_val, epoch_m), grads = Flux.withgradient(model) do m
                pred_angles = vec(m(x_5d))
                reconstructed = diff_rotate_3d(img_3d, pred_angles)
                mse = sum((reconstructed .- imagePrim) .^ 2) / length(imagePrim)
                mae = sum(abs.(reconstructed .- imagePrim)) / length(imagePrim)
                return mse, mae
            end
            Flux.update!(opt_state, model, grads[1])
            epoch_mse += loss_val
            epoch_mae += epoch_m
        end
        avg_train_mse = epoch_mse / N_TRAIN
        avg_train_mae = epoch_mae / N_TRAIN
        
        # Compute endpoints and test metrics for visualization
        x_5d_1 = test_imgs[:,:,:,:,1:1]
        img_3d_1 = test_imgs[:,:,:,1,1]
        pred_angles_1 = vec(model(x_5d_1))
        reconstructed_1 = diff_rotate_3d(img_3d_1, pred_angles_1)
        p1, p2 = extract_endpoints(reconstructed_1)
        push!(endpoints_history, (epoch, p1, p2))

        test_mse = 0.0f0
        test_mae = 0.0f0
        for i in 1:N_TEST
            x_5d = test_imgs[:,:,:,:,i:i]
            img_3d = test_imgs[:,:,:,1,i]
            m_mse, m_mae = compute_metrics(model, x_5d, img_3d, imagePrim)
            test_mse += m_mse
            test_mae += m_mae
        end
        avg_test_mse = test_mse / N_TEST
        avg_test_mae = test_mae / N_TEST
        push!(history, (epoch, avg_train_mse, avg_train_mae, avg_test_mse, avg_test_mae))
        
        if epoch == 1 || epoch % 10 == 0 || epoch == EPOCHS
            @printf("  Epoch %-4d | Train MSE: %.6f | Test MSE: %.6f | Test MAE: %.6f\n", epoch, avg_train_mse, avg_test_mse, avg_test_mae)
        end
    end

    # Save reconstructed.nii.gz for final epoch
    x_5d_1 = test_imgs[:,:,:,:,1:1]
    img_3d_1 = test_imgs[:,:,:,1,1]
    pred_angles_1 = vec(model(x_5d_1))
    reconstructed_final = diff_rotate_3d(img_3d_1, pred_angles_1)
    MedImages.Load_and_save.create_nii_from_medimage(make_medimage(reconstructed_final), joinpath(OUTPUT_ARTIFACT_DIR, "reconstructed.nii.gz"))

    # Save history
    println("\n[4/4] Saving history...")
    open(joinpath(OUTPUT_ARTIFACT_DIR, "loss_history.csv"), "w") do io
        println(io, "epoch,train_mse,train_mae,test_mse,test_mae")
        for (e, tr_mse, tr_mae, ts_mse, ts_mae) in history
            println(io, "$e,$tr_mse,$tr_mae,$ts_mse,$ts_mae")
        end
    end
    
    open(joinpath(OUTPUT_ARTIFACT_DIR, "endpoints.csv"), "w") do io
        println(io, "epoch,p1x,p1y,p1z,p2x,p2y,p2z")
        for (epoch, p1, p2) in endpoints_history
            println(io, "$epoch,$(p1[1]),$(p1[2]),$(p1[3]),$(p2[1]),$(p2[2]),$(p2[3])")
        end
    end
    println("Done.")
end

main()
