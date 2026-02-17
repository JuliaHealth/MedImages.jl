println("Script starting: Loading dependencies...")
flush(stdout)
using MedImages
using MedImages.MedImage_data_struct
using MedImages.Utils
using MedImages.Basic_transformations
using Lux
using Zygote
using Optimisers
using Functors
using Random
using LinearAlgebra
using Test
using CUDA

# 1. Device Selection
# Manual GPU detection without LuxCUDA
use_gpu = CUDA.functional()
println("CUDA functional: $use_gpu")
if use_gpu
    println("Using GPU device: $(CUDA.device())")
end

# 2. Synthetic Cube Generator
function create_cube_medimage(size_t=(8,8,8), cube_size=4)
    data = zeros(Float32, size_t)
    cx, cy, cz = size_t .÷ 2
    hs = cube_size ÷ 2
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

# 3. Model Definition (Lux)
function create_model()
    return Chain(
        FlattenLayer(),
        Dense(8*8*8, 64, relu),
        Dense(64, 12) 
    )
end

# 4. Affine Parameter mapping
function params_to_matrix(p)
    sc = (1.0f0 + p[1], 1.0f0 + p[2], 1.0f0 + p[3])
    rot = (p[4], p[5], p[6])
    tr = (p[7], p[8], p[9])
    sh = (p[10], p[11], p[12])
    
    return create_affine_matrix(translation=tr, rotation=rot, scale=sc, shear=sh)
end

# 5. Helper for device migration
to_dev(x) = use_gpu ? cu(x) : x

# 6. Training Loop
function train()
    # Hybrid Strategy:
    # 1. Model and parameter mapping stay on CPU for stability with Zygote/Enzyme.
    # 2. Voxel data stays on GPU to ensure affine_transform_mi runs on GPU.
    # 3. Gradients flow correctly through the device boundaries via Zygote's Array/CuArray rules.
    
    base_cube = create_cube_medimage()
    
    model = create_model()
    ps, st = Lux.setup(Random.default_rng(), model)
    
    # ps, st stay on CPU
    
    # Use Adam optimizer
    opt = Adam(0.0005)
    tstate = Optimisers.setup(opt, ps)
    
    batch_size = 2
    
    # Target: slight rotation and translation
    target_params_raw = [randn(Float32, 12) .* 0.03f0 for _ in 1:batch_size]
    target_matrices = [params_to_matrix(tp) for tp in target_params_raw]
    
    # Create batch and move to GPU
    batch_base = create_batched_medimage([base_cube for _ in 1:batch_size])
    if use_gpu
        batch_base.voxel_data = cu(batch_base.voxel_data)
    end
    
    # Target batch generated on GPU
    println("Generating target images on GPU: $use_gpu")
    target_batch = affine_transform_mi(batch_base, target_matrices, Linear_en)
    
    # Move a CPU-compatible copy of base voxel data for the model forward pass
    # (since the model is on CPU)
    voxel_data_cpu = Array(batch_base.voxel_data)
    
    initial_loss = 0.0
    final_loss = 0.0

    function loss_fn(p)
        # Lux.apply on CPU
        preds, _ = Lux.apply(model, voxel_data_cpu, p, st)
        
        # Mapping predicted params to matrices (on CPU)
        pred_matrices = [params_to_matrix(preds[:, b]) for b in 1:batch_size]
        
        # TRANSFORMATION RUNS ON GPU (if batch_base is on GPU)
        reconstructed = affine_transform_mi(batch_base, pred_matrices, Linear_en)
        
        # MSE Loss - Zygote handles CuArray subtraction and sum correctly
        loss = sum((reconstructed.voxel_data .- target_batch.voxel_data).^2)
        return loss
    end

    n_iters = 50
    println("Starting training for $n_iters iterations...")
    for i in 1:n_iters
        l, back = Zygote.pullback(loss_fn, ps)
        if i == 1
            initial_loss = l
        end
        final_loss = l
        
        if i % 10 == 1 || i == n_iters
             println("Iter $i: Loss = $l")
             flush(stdout)
        end
        
        gs = back(1.0f0)[1]
        tstate, ps = Optimisers.update(tstate, ps, gs)
    end
    
    println("\nSummary:")
    println("Initial Loss: $initial_loss")
    println("Final Loss:   $final_loss")
    
    if final_loss < initial_loss
        println("SUCCESS: Loss function value dropped.")
    else
        println("FAILURE: Loss function value did not drop.")
    end
    
    @test final_loss < initial_loss
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
