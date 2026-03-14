using MPI
MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const world_size = MPI.Comm_size(comm)

using Lux, Random, Optimisers, Zygote, Statistics
using KernelAbstractions
using MedImages
using MedImages.MedImage_data_struct
using Printf
using Dates
using Serialization
using Wandb
using Accessors
using HDF5
using Functors: fmap

# Optional: Distributed Utils
# We check if we are running in distributed mode
const DISTRIBUTED_MODE = MPI.Initialized() ? true : false

# If distributed, we might want to use CUDA if available on local rank
# For now, we stick to CPU or basic GPU detection.

# Ensure local imports work relative to this file
if !@isdefined(Preprocessing)
    include(joinpath(@__DIR__, "src", "preprocessing.jl"))
end
if !@isdefined(RegistrationModel)
    include(joinpath(@__DIR__, "src", "model.jl"))
end
if !@isdefined(FusedLoss)
    include(joinpath(@__DIR__, "src", "fused_loss.jl"))
end

if !@isdefined(RegistrationUtils)
    include(joinpath(@__DIR__, "src", "utils.jl"))
end

using .Preprocessing
using .RegistrationModel
using .FusedLoss

# --- Distributed Helper (AllReduce) ---
function allreduce_recursive!!(backend, grads)
    # Recursively walk through gradients and Allreduce leaf arrays
    # fmap allows us to apply a function to all leaves of a Lux parameter/gradient structure
    return fmap(grads) do leaf
        if leaf isa AbstractArray
            # Perform in-place allreduce on the buffer
            # We use MPI.Allreduce! 
            MPI.Allreduce!(leaf, MPI.SUM, MPI.COMM_WORLD)
            # Average by world size
            leaf ./= Float32(MPI.Comm_size(MPI.COMM_WORLD))
        end
        return leaf
    end
end
function visualize_validation(epoch, model, ps, st, valid_train, f, device, ka_backend, rank, dataset_path, num_organs)
    if rank != 0 return end
    
    println("--- Running Visualization for Epoch $epoch ---")
    data_root = joinpath("experiments", "organ_affine_registration", "data")
    output_dir = joinpath(data_root, "epoch_$epoch")
    mkpath(output_dir)
    
    # Use first patient in train as Atlas Reference
    if isempty(valid_train)
        @warn "No valid training subjects for visualization reference."
        return
    end
    ref_pat = valid_train[1]
    ref_gold = read(f[ref_pat]["gold"]) # (X,Y,Z,N)
    sx, sy, sz, _ = size(ref_gold)
    
    # Prepare Atlas Channels on Device
    atlas_channels = KernelAbstractions.allocate(ka_backend, Float32, sx, sy, sz, 1, num_organs)
    for i in 1:num_organs
        copyto!(view(atlas_channels, :, :, :, 1, i), ref_gold[:,:,:,i])
    end
    
    # Select 2 validation subjects
    val_list = read(f["val_list"])
    valid_val = String[]
    for p in val_list
        if haskey(f, p)
            pat_grp = f[p]
            if (haskey(pat_grp, "image") || haskey(pat_grp, "spect")) && haskey(pat_grp, "gold")
                push!(valid_val, p)
            end
        end
    end
    
    if isempty(valid_val)
        @warn "No valid subjects for visualization."
        return
    end
    
    targets = valid_val[1:min(2, length(valid_val))]
    labels = [1, 2, 3, 6] 
    
    for pat_id in targets
        # Load patient data
        pat_grp = f[pat_id]
        spect_raw = haskey(pat_grp, "image") ? read(pat_grp["image"]) : read(pat_grp["spect"])
        atlas_raw = read(f["atlas_image"])
        gold_raw = read(pat_grp["gold"])
        
        # Normalize for input
        spect_norm = Float32.(spect_raw) ./ 1000f0
        atlas_norm = Float32.(atlas_raw) ./ 1000f0
        
        # Prepare Input
        x_raw = cat(reshape(atlas_norm, size(atlas_norm)..., 1), 
                     reshape(spect_norm, size(spect_norm)..., 1), dims=4)
        x_raw = reshape(x_raw, size(x_raw)..., 1) |> device
        
        # Predict Parameters
        params_pred, _ = model(x_raw, ps, st)
        ap_cpu = Array(reshape(params_pred, 15, num_organs))
        
        # Invert and Warp
        warped_channels = KernelAbstractions.zeros(ka_backend, Float32, sx, sy, sz, 1, num_organs)
        inv_matrices = zeros(Float32, 3, 4, num_organs)
        for i in 1:num_organs
            cx_v, cy_v, cz_v = ap_cpu[13, i], ap_cpu[14, i], ap_cpu[15, i]
            M = params_to_matrix(ap_cpu[:, i], [cx_v, cy_v, cz_v])
            M_inv = inv(M)
            inv_matrices[:, :, i] .= Float32.(M_inv[1:3, :])
        end
        inv_matrices_gpu = KernelAbstractions.allocate(ka_backend, Float32, size(inv_matrices))
        copyto!(inv_matrices_gpu, inv_matrices)
        dummy_centers_gpu = KernelAbstractions.zeros(ka_backend, Float32, 3, num_organs)
        warp_kernel = FusedLoss.gpu_batch_affine_warp_kernel!(ka_backend, (8, 8, 8))
        warp_kernel(warped_channels, atlas_channels, inv_matrices_gpu, dummy_centers_gpu, ndrange=size(warped_channels))
        KernelAbstractions.synchronize(ka_backend)
        
        # Fuse labels for prediction
        warped_cpu = Array(warped_channels)
        final_seg = zeros(Float32, sx, sy, sz)
        gold_fused = zeros(Float32, sx, sy, sz)
        for i in 1:num_organs
            final_seg .+= warped_cpu[:,:,:,1,i] .* Float32(labels[i])
            gold_fused .+= Float32.(gold_raw[:,:,:,i]) .* Float32(labels[i])
        end
        
        # Save NIfTIs
        pat_dir = joinpath(output_dir, pat_id)
        mkpath(pat_dir)
        
        attrs_grp = HDF5.attributes(pat_grp)
        spacing = haskey(attrs_grp, "spacing") ? Tuple(read(attrs_grp["spacing"])) : (1.0f0, 1.0f0, 1.0f0)
        origin = haskey(attrs_grp, "origin") ? Tuple(read(attrs_grp["origin"])) : (0.0f0, 0.0f0, 0.0f0)
        direction = haskey(attrs_grp, "direction") ? Tuple(read(attrs_grp["direction"])) : (1.0f0, 0.0f0, 0.0f0, 0.0f0, 1.0f0, 0.0f0, 0.0f0, 0.0f0, 1.0f0)
        
        mi_base = MedImage(
            voxel_data=final_seg, 
            origin=origin, 
            spacing=spacing, 
            direction=direction,
            image_type=MedImage_data_struct.CT_type,
            image_subtype=MedImage_data_struct.CT_subtype,
            patient_id=pat_id
        )
        
        # 1. Registered Atlas Segmentation (Output)
        MedImages.Load_and_save.create_nii_from_medimage(mi_base, joinpath(pat_dir, "registered_atlas_seg.nii.gz"))
        
        # 2. Atlas Input (Moving image)
        MedImages.Load_and_save.create_nii_from_medimage(@set(mi_base.voxel_data = Float32.(atlas_raw)), joinpath(pat_dir, "atlas_input.nii.gz"))
        
        # 3. SPECT Input (Fixed image / nukl)
        MedImages.Load_and_save.create_nii_from_medimage(@set(mi_base.voxel_data = Float32.(spect_raw)), joinpath(pat_dir, "spect_input.nii.gz"))
        
        # 4. Gold Standard (Target segmentation)
        MedImages.Load_and_save.create_nii_from_medimage(@set(mi_base.voxel_data = gold_fused), joinpath(pat_dir, "gold_standard.nii.gz"))
        
        # Copy original files
        try
            src_pat_path = joinpath(dataset_path, pat_id)
            if isdir(src_pat_path)
                ct_path = joinpath(src_pat_path, "SPECT_DATA", "CT.nii.gz")
                if isfile(ct_path) cp(ct_path, joinpath(pat_dir, "CT.nii.gz"), force=true) end
            end
        catch e
            @warn "Failed to copy CT for $pat_id: $e"
        end
        
        println("Saved visualization for $pat_id")
    end
end

# --- Overfit Experiment ---
function run_training_experiment(args=ARGS)
    # Initialize MPI if needed
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)

    rng = Random.default_rng()
    Random.seed!(rng, 1234 + rank)

    if rank == 0
        println("--- Starting Organ Affine Registration Training ---")
        println("MPI World Size: $world_size")
    end

    # Initialize WandB
    lg = nothing
    if rank == 0
        lg = WandbLogger(project = "organ_affine_registration",
                         name = "train_4gpu_wandb_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))",
                         config = Dict("num_organs" => 4, "world_size" => world_size, "learning_rate" => 0.0001))
    end

    # 1. HDF5 Data Loading
    hdf5_path = "/mnt/vast-kisski/projects/ovgu_medicine_llm/ollama_data/dataset_unified_for_registration.h5"
    if !isfile(hdf5_path)
        if rank == 0 println("ERROR: HDF5 file not found. Run preprocess_data.jl first.") end
        return
    end
    gold_vol = zeros(Float32, 1, 1, 1, 1) # Placeholder for scope if needed

    f = h5open(hdf5_path, "r")
    train_list = read(f["train_list"])
    valid_train = String[]
    for p in train_list
        if haskey(f, p)
            pat_grp = f[p]
            if (haskey(pat_grp, "image") || haskey(pat_grp, "spect")) && haskey(pat_grp, "gold")
                push!(valid_train, p)
            end
        end
    end
    
    if rank == 0 println("Valid training subjects (filtered): $(length(valid_train))") end

    # 2. Model Setup
    num_organs = 4
    model = MultiScaleCNN(2, num_organs)
    
    ps, st = Lux.setup(rng, model)

    opt = Optimisers.Adam(0.0001)
    opt_state = Optimisers.setup(opt, ps)

    # 3. Atlas Points (From one valid patient)
    atlas_points_cpu = fill(-1.0f0, (3, 512, num_organs))
    if !isempty(valid_train)
        first_pat = valid_train[1]
        atlas_vol = read(f["atlas_image"])
        gold_vol = read(f[first_pat]["gold"]) # (X,Y,Z,N)
        for i in 1:num_organs
            idxs = findall(gold_vol[:, :, :, i] .> 0.5f0)
            if !isempty(idxs)
                n = length(idxs)
                sel = round.(Int, range(1, stop=n, length=512))
                subset = idxs[sel]
                for (j, idx) in enumerate(subset)
                    atlas_points_cpu[1, j, i] = Float32(idx[1])
                    atlas_points_cpu[2, j, i] = Float32(idx[2])
                    atlas_points_cpu[3, j, i] = Float32(idx[3])
                end
            end
        end
        atlas_vol = nothing  # free after extracting points
        gold_vol = nothing
        GC.gc()
    end
    atlas_points = atlas_points_cpu |> device

    # 4. Training Loop
    backend = KernelAbstractions.get_backend(ps.dense_layers.layer_1.weight)
    device = backend isa KernelAbstractions.GPU ? Lux.gpu_device() : Lux.cpu_device()
    
    # Pre-read Atlas Image once, then free the raw buffer
    hp_atlas = read(f["atlas_image"])
    atlas_norm = Float32.(hp_atlas) ./ 1000f0
    hp_atlas = nothing  # free raw buffer immediately
    GC.gc()

    # --- Checkpoint Loading ---
    start_epoch = 1
    checkpoint_path = "checkpoint_latest.jls"
    if isfile(checkpoint_path)
        if rank == 0 println("--- Loading Checkpoint from $checkpoint_path ---") end
        cp_data = deserialize(checkpoint_path)
        ps = cp_data.ps |> device
        st = cp_data.st |> device
        opt_state = cp_data.opt_state
        start_epoch = cp_data.epoch + 1
        cp_data = nothing
        GC.gc()
    end

    max_epochs = 500
    for epoch in start_epoch:max_epochs
        shuffle!(rng, valid_train)
        
        for pat_id in valid_train
            # Load Data
            if !haskey(f, pat_id) continue end
            pat_grp = f[pat_id]
            
            # Robust Key Access
            if (!haskey(pat_grp, "image") && !haskey(pat_grp, "spect")) || !haskey(pat_grp, "gold")
                @warn "Patient $pat_id missing required keys (image/spect or gold). Skipping."
                continue
            end
            hp_spect = haskey(pat_grp, "image") ? read(pat_grp["image"]) : read(pat_grp["spect"])
            hp_gold = read(pat_grp["gold"])
            
            # 2x Downsample (Strided) to save 8x memory
            # Use views if possible to avoid copying, but here we need a new array for now
            spect = hp_spect[1:2:end, 1:2:end, 1:2:end]
            atlas = atlas_norm[1:2:end, 1:2:end, 1:2:end]
            gold = hp_gold[1:2:end, 1:2:end, 1:2:end, :]
            
            # Normalize SPECT
            spect = Float32.(spect) ./ 1000f0
            
            x_raw = cat(reshape(atlas, size(atlas)..., 1), 
                         reshape(spect, size(spect)..., 1), dims=4)
            x_raw = reshape(x_raw, size(x_raw)..., 1) |> device
            gold_raw = reshape(gold, size(gold)..., 1) |> device
            
            # Loss Metadata
            bary_raw = read(pat_grp["barycenters"]) ./ 2.0f0
            radii_raw = read(pat_grp["radii"]) ./ 2.0f0
            
            bary_gpu = bary_raw |> device
            radii_gpu = radii_raw |> device
            
            # Augmentation
            sx, sy, sz = size(atlas)
            centers = Float32[sx/2; sy/2; sz/2]
            centers = reshape(centers, 3, 1)
            
            x_aug, gold_aug, am_meta = Zygote.ignore() do
                apply_random_augmentation(x_raw, gold_raw, centers, backend)
            end

            # Gradient Step
            (l, st), back = Zygote.pullback(p -> begin
                params_pred, new_state = model(x_aug, p, st)
                ap = reshape(params_pred, 15, num_organs, 1)
                l_val = compute_organ_loss(atlas_points, ap, gold_aug[:,:,:,:,1], bary_gpu, radii_gpu)
                return l_val, new_state
            end, ps)
            
            if isnan(l) || isinf(l)
                @warn "Invalid loss (NaN/Inf) for Pat $pat_id. Skipping update."
                continue
            end

            grads = back((1.0f0, nothing))[1]
            if any(isnan, grads.dense_layers.layer_4.weight)
                 @warn "Gradients are NaN for Pat $pat_id. Skipping update."
                 continue
            end

            grads = allreduce_recursive!!(backend, grads)
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            
            if rank == 0 
                @printf("Epoch %d | Pat %s | Loss = %.5f\n", epoch, pat_id, l)
                Wandb.log(lg, Dict("loss" => l, "epoch" => epoch))
            end
            
            # Memory Management
            x_raw = nothing
            gold_raw = nothing
            x_aug = nothing
            gold_aug = nothing
            hp_spect = nothing
            hp_gold = nothing
            bary_raw = nothing
            radii_raw = nothing
            bary_gpu = nothing
            radii_gpu = nothing
            GC.gc()
            if backend isa KernelAbstractions.GPU
                # CUDA.reclaim() # Only if using CUDA.jl directly
            end
        end
        
        # --- End of Epoch: Saving and Visualization ---
        if rank == 0
            println("--- Epoch $epoch Summary ---")
            # Save Checkpoint (on CPU to be safe)
            checkpoint_data = (ps = ps |> Lux.cpu_device(), st = st |> Lux.cpu_device(), 
                               opt_state = opt_state, epoch = epoch)
            serialize(checkpoint_path, checkpoint_data)
            if epoch % 10 == 0
                serialize("checkpoint_epoch_$epoch.jls", checkpoint_data)
            end
            println("--- Saved Checkpoint for Epoch $epoch ---")
        end

        dataset_path = "/mnt/vast-kisski/projects/ovgu_medicine_llm/ollama_data/dataset_Lu"
        visualize_validation(epoch, model, ps, st, valid_train, f, device, backend, rank, dataset_path, num_organs)
    end
    
    close(f)
    if rank == 0
        close(lg)
    end
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    run_training_experiment()
end
