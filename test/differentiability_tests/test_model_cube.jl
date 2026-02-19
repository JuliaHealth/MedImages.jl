println("Script starting: Loading MedImages...")
flush(stdout)
using MedImages
println("MedImages loaded. Loading other dependencies...")
flush(stdout)
using MedImages.MedImage_data_struct
using MedImages.Utils
using MedImages.Basic_transformations
using Lux
using Zygote
using Optimisers
using Random
using LinearAlgebra
using Test

# 1. Synthetic Cube Generator
function create_cube_medimage(size_t=(8,8,8), cube_size=4)
    data = zeros(Float32, size_t)
    cx, cy, cz = size_t .÷ 2
    hs = cube_size ÷ 2
    # Ensure indices are within bounds
    x_range = max(1, cx-hs):min(size_t[1], cx+hs)
    y_range = max(1, cy-hs):min(size_t[2], cy+hs)
    z_range = max(1, cz-hs):min(size_t[3], cz+hs)
    data[x_range, y_range, z_range] .= 1.0f0
    
    return MedImage(
        voxel_data = data,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        image_type = MedImages.MedImage_data_struct.MRI_type,
        image_subtype = MedImages.MedImage_data_struct.T1_subtype,
        patient_id = "cube"
    )
end

# 2. Model Definition (Lux)
function create_model()
    return Chain(
        FlattenLayer(),
        Dense(8*8*8, 32, relu),
        Dense(32, 12) 
    )
end

# 3. Affine Parameter mapping
function params_to_matrix(p)
    sc = (1.0f0 + p[1], 1.0f0 + p[2], 1.0f0 + p[3])
    rot = (p[4], p[5], p[6])
    tr = (p[7], p[8], p[9])
    sh = (p[10], p[11], p[12])
    
    return create_affine_matrix(translation=tr, rotation=rot, scale=sc, shear=sh)
end

# 4. Training Loop
function train()
    # Random.seed!(42)
    base_cube = create_cube_medimage()
    
    model = create_model()
    ps, st = Lux.setup(Random.default_rng(), model)
    
    opt = Adam(0.01)
    tstate = Optimisers.setup(opt, ps)
    
    batch_size = 1 # Ultra small
    
    target_params = [randn(Float32, 12) .* 0.02f0 for _ in 1:batch_size]
    target_matrices = [params_to_matrix(tp) for tp in target_params]
    batch_base = create_batched_medimage([base_cube for _ in 1:batch_size])
    
    println("Generating target images (8x8x8)...")
    target_batch = affine_transform_mi(batch_base, target_matrices, Linear_en)
    
    function loss_fn(p)
        println("  Forward pass...")
        voxel_data = target_batch.voxel_data
        preds, _ = Lux.apply(model, voxel_data, p, st)
        
        pred_matrices = [params_to_matrix(preds[:, b]) for b in 1:batch_size]
        reconstructed = affine_transform_mi(batch_base, pred_matrices, Linear_en)
        
        loss = sum((reconstructed.voxel_data .- target_batch.voxel_data).^2)
        println("  Loss: $loss")
        return loss
    end

    println("Starting image reconstruction training (8x8x8)...")
    for i in 1:10
        println("Iter $i starting...")
        flush(stdout)

        l, back = Zygote.pullback(loss_fn, ps)
        println("  Backward pass...")
        flush(stdout)
        
        gs = back(1.0)[1]
        
        if gs === nothing
             println("WARNING: Gradients are nothing at Iter $i.")
        end
        
        tstate, ps = Optimisers.update(tstate, ps, gs)
        println("Iter $i finished. Loss: $l")
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
