
using Lux, Optimisers, Zygote, Random, Statistics, Dates
using MedImages
using MedImages.Basic_transformations
using MedImages.MedImage_data_struct
using MedImages.Utils: create_batched_medimage
using LinearAlgebra

# 1. Synthetic Data Generation
function generate_synthetic_data(batch_size, spatial_size=(32, 32, 32))
    # Create a primitive image: a line or bar
    imagePrim = zeros(Float32, spatial_size)

    # Create a bar along Z axis in the center
    cx, cy = spatial_size[1] ÷ 2, spatial_size[2] ÷ 2
    r = 4
    imagePrim[cx-r:cx+r, cy-r:cy+r, :] .= 1.0f0

    # Generate random angles for the batch
    rng = Random.default_rng()
    angles_x = (rand(rng, Float64, batch_size) .- 0.5) .* 90.0 # -45 to 45 deg
    angles_y = (rand(rng, Float64, batch_size) .- 0.5) .* 90.0
    angles_z = (rand(rng, Float64, batch_size) .- 0.5) .* 90.0

    # Create base BatchedMedImage
    voxel_data_base = repeat(imagePrim, 1, 1, 1, batch_size)

    # Metadata
    mri_type = MedImages.MedImage_data_struct.MRI_type
    subtype = MedImages.MedImage_data_struct.T1_subtype
    date_val = Dates.now()

    med_imgs = [MedImage(
        voxel_data=imagePrim,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0,0.0,0.0),
        direction=Tuple(vec(Matrix(1.0I,3,3))),
        image_type=mri_type,
        image_subtype=subtype,
        date_of_saving=date_val,
        acquistion_time=date_val,
        patient_id="synth"
        ) for _ in 1:batch_size]
    batch_base = create_batched_medimage(med_imgs)

    # Create ground truth rotation matrices
    matrices_gt = Vector{Matrix{Float64}}(undef, batch_size)
    for b in 1:batch_size
        Rx = Rodrigues_rotation_matrix(batch_base.direction[b], 1, angles_x[b])
        Ry = Rodrigues_rotation_matrix(batch_base.direction[b], 2, angles_y[b])
        Rz = Rodrigues_rotation_matrix(batch_base.direction[b], 3, angles_z[b])

        R_comb = Rz * Ry * Rx

        M = Matrix{Float64}(I, 4, 4)
        M[1:3, 1:3] = R_comb
        matrices_gt[b] = M
    end

    # Apply GT rotation to create X
    # We use Linear interpolation for generation
    batch_X = affine_transform_mi(batch_base, matrices_gt, MedImages.MedImage_data_struct.Linear_en)

    # Target Y is the original unrotated batch
    batch_Y = batch_base

    return batch_X, batch_Y, (angles_x, angles_y, angles_z)
end

# 2. Model Definition
function create_model()
    return Chain(
        # Input reshape: (X, Y, Z, 1, Batch)
        WrappedFunction(x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), 1, size(x, 4))),

        Conv((5, 5, 5), 1 => 8, relu; pad=2),
        MaxPool((2, 2, 2)),

        Conv((3, 3, 3), 8 => 16, relu; pad=1),
        MaxPool((2, 2, 2)),

        FlattenLayer(),

        Dense(16 * 8 * 8 * 8 => 64, relu),
        Dense(64 => 3) # Output: 3 Euler angles
    )
end

# 3. Helper to build matrix from angles (Differentiable)
function angles_to_matrices(angles_pred_batch, directions)
    batch_size = size(angles_pred_batch, 2)

    # Zygote-friendly map
    matrices = map(1:batch_size) do b
        ax = angles_pred_batch[1, b]
        ay = angles_pred_batch[2, b]
        az = angles_pred_batch[3, b]
        dir = directions[b]

        Rx = Rodrigues_rotation_matrix(dir, 1, Float64(ax))
        Ry = Rodrigues_rotation_matrix(dir, 2, Float64(ay))
        Rz = Rodrigues_rotation_matrix(dir, 3, Float64(az))

        R = Rz * Ry * Rx

        M = Matrix{Float64}(I, 4, 4)
        M[1:3, 1:3] = R
        M
    end
    return matrices
end

# Helper: Pure Julia Differentiable Grid Generator
function generate_grid_differentiable(matrices, spatial_size)
    batch_size = length(matrices)
    w, h, d = spatial_size
    n_points = w * h * d

    cx = (w + 0.0) / 2.0
    cy = (h + 0.0) / 2.0
    cz = (d + 0.0) / 2.0

    # Construct base coordinates
    xyz = zeros(Float32, 4, n_points)
    idx = 1
    for k in 1:d, j in 1:h, i in 1:w
        xyz[1, idx] = i - cx
        xyz[2, idx] = j - cy
        xyz[3, idx] = k - cz
        xyz[4, idx] = 1.0
        idx += 1
    end

    # Apply matrices (batch-wise)
    points_vec = map(matrices) do M
        M_inv = inv(M)
        p = M_inv * xyz

        # Shift back
        p[1, :] .+= cx
        p[2, :] .+= cy
        p[3, :] .+= cz

        p[1:3, :]
    end

    return stack(points_vec, dims=3)
end

# 4. Loss Function
function compute_loss(model, ps, st, x_batch, y_target_voxel_data, directions)
    input_data = x_batch.voxel_data

    angles_pred, st_new = model(input_data, ps, st)

    # Construct matrices from predicted angles
    matrices = angles_to_matrices(angles_pred, directions)

    # Generate interpolation grid (Differentiable)
    points = generate_grid_differentiable(matrices, size(input_data)[1:3])

    # Interpolate using MedImages core function
    # interpolate_pure(points, input, spacing, keep_beginning, extrapolate, is_nn)
    resampled = MedImages.Utils.interpolate_pure(points, input_data, x_batch.spacing, false, 0.0, false)

    # Reshape
    sz = size(input_data)
    output = reshape(resampled, sz[1], sz[2], sz[3], sz[4])

    # MSE Loss
    l = sum(abs2, output .- y_target_voxel_data) / length(output)

    return l, st_new
end

# 5. Main Test
function run_test()
    println("--- Lux Spatial Transformer Test ---")
    println("Generating Synthetic Data (Rotated Lines)...")
    batch_size = 4
    X, Y, angles_gt = generate_synthetic_data(batch_size)

    println("Initializing Lux Model...")
    model = create_model()
    ps, st = Lux.setup(Random.default_rng(), model)

    opt = Optimisers.Adam(0.1)
    opt_state = Optimisers.setup(opt, ps)

    println("Training for 20 iterations...")
    losses = []

    directions = X.direction

    for i in 1:20
        (l, st), back = Zygote.pullback(p -> compute_loss(model, p, st, X, Y.voxel_data, directions), ps)
        grads = back((1.0f0, nothing))[1]

        opt_state, ps = Optimisers.update(opt_state, ps, grads)

        push!(losses, l)
        println("Iter $i: Loss = $l")
    end

    println("Initial Loss: $(losses[1])")
    println("Final Loss: $(losses[end])")

    if losses[end] < losses[1]
        println("SUCCESS: Loss decreased significantly.")
    else
        println("FAILURE: Loss did not decrease.")
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_test()
end
