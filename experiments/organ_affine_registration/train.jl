using Lux, Random, Optimisers, Zygote, Statistics
using KernelAbstractions, LuxCUDA, HDF5
using MedImages
using MedImages.MedImage_data_struct
using Printf
using MPI

# Ensure local imports
if !@isdefined(Preprocessing)
    include("src/preprocessing.jl")
end
if !@isdefined(RegistrationModel)
    include("src/model.jl")
end
if !@isdefined(FusedLoss)
    include("src/fused_loss.jl")
end

if !@isdefined(RegistrationUtils)
    include("src/utils.jl")
end

using .Preprocessing
using .RegistrationModel
using .FusedLoss
using .RegistrationUtils

# --- Visualization / Inference Helpers ---
# Moved to src/utils.jl

function visualize_validation(epoch, model, ps, st, valid_train, f, device, ka_backend, rank, dataset_path, num_organs)
    if rank != 0 return end
    
    println("--- Running Visualization for Epoch $epoch ---")
    output_dir = "epoch_$epoch"
    mkpath(output_dir)
    
    # Use first patient in train as Atlas Reference
    ref_pat = valid_train[1]
    ref_gold = read(f[ref_pat]["gold"]) # (X,Y,Z,N)
    sx, sy, sz, _ = size(ref_gold)
    
    # Prepare Atlas Channels on Device
    atlas_channels = KernelAbstractions.allocate(ka_backend, Float32, sx, sy, sz, 1, num_organs)
    for i in 1:num_organs
        ch = reshape(ref_gold[:,:,:,i], sx, sy, sz, 1, 1)
        # copyto! expects same size or specific subarray
        copyto!(view(atlas_channels, :, :, :, 1, i), ref_gold[:,:,:,i])
    end
    
    # Select 2 validation subjects
    val_list = read(f["val_list"])
    valid_val = [p for p in val_list if p in keys(f)]
    if isempty(valid_val) return end
    
    targets = valid_val[1:min(2, length(valid_val))]
    labels = [1, 2, 3, 6] # Organs matching SELECTED_ORGANS
    
    for pat_id in targets
        # Load patient data (full resolution for inference visualization)
        pat_grp = f[pat_id]
        spect = read(pat_grp["spect"])
        atlas = read(pat_grp["atlas"])
        
        # Prepare Input
        x_raw = cat(reshape(atlas, size(atlas)..., 1), 
                     reshape(spect, size(spect)..., 1), dims=4)
        x_raw = reshape(x_raw, size(x_raw)..., 1) |> device
        
        # Predict Parameters
        params_pred, _ = model(x_raw, ps, st)
        ap_cpu = Array(reshape(params_pred, 15, num_organs))
        
        # Invert and Warp
        warped_sum = zeros(Float32, sx, sy, sz)
        warped_channels = KernelAbstractions.zeros(ka_backend, Float32, sx, sy, sz, 1, num_organs)
        
        # We need T^-1 (Patient -> Atlas)
        inv_matrices = zeros(Float32, 3, 4, num_organs)
        for i in 1:num_organs
            # Get center from metadata if possible, else middle
            cx_v, cy_v, cz_v = ap_cpu[13, i], ap_cpu[14, i], ap_cpu[15, i]
            M = params_to_matrix(ap_cpu[:, i], [cx_v, cy_v, cz_v])
            M_inv = inv(M)
            inv_matrices[:, :, i] .= Float32.(M_inv[1:3, :])
        end
        
        # Move to GPU
        inv_matrices_gpu = KernelAbstractions.allocate(ka_backend, Float32, size(inv_matrices))
        copyto!(inv_matrices_gpu, inv_matrices)
        
        # Dummy centers for the warp kernel since we already shifted back in Matrix
        dummy_centers_gpu = KernelAbstractions.zeros(ka_backend, Float32, 3, num_organs)
        
        # Launch Warp
        warp_kernel = FusedLoss.gpu_batch_affine_warp_kernel!(ka_backend, (8, 8, 8))
        warp_kernel(warped_channels, atlas_channels, inv_matrices_gpu, dummy_centers_gpu, ndrange=size(warped_channels))
        KernelAbstractions.synchronize(ka_backend)
        
        # Fuse labels
        warped_cpu = Array(warped_channels)
        final_seg = zeros(Float32, sx, sy, sz)
        for i in 1:num_organs
            final_seg .+= warped_cpu[:,:,:,1,i] .* Float32(labels[i])
        end
        
        # Save NIfTI
        pat_dir = joinpath(output_dir, pat_id)
        mkpath(pat_dir)
        
        # Metadata from attributes
        spacing = Tuple(read(HDF5.attributes(pat_grp)["spacing"]))
        origin = Tuple(read(HDF5.attributes(pat_grp)["origin"]))
        direction = Tuple(read(HDF5.attributes(pat_grp)["direction"]))
        
        mi = MedImage(
            voxel_data=final_seg, 
            origin=origin, 
            spacing=spacing, 
            direction=direction,
            image_type=MedImage_data_struct.CT_type,
            image_subtype=MedImage_data_struct.CT_subtype,
            patient_id=pat_id
        )
        MedImages.Load_and_save.create_nii_from_medimage(mi, joinpath(pat_dir, "registered_atlas_seg.nii.gz"))
        
        # Copy original files
        src_pat_path = joinpath(dataset_path, pat_id)
        cp(joinpath(src_pat_path, "SPECT_DATA", "CT.nii.gz"), joinpath(pat_dir, "CT.nii.gz"), force=true)
        cp(joinpath(src_pat_path, "Atlas", "Atlas_Registered.nii.gz"), joinpath(pat_dir, "Atlas_Registered.nii.gz"), force=true)
        cp(joinpath(src_pat_path, "TOTAL_SEGMENTOR_OUTPUT", "segmentation.nii.gz"), joinpath(pat_dir, "segmentation.nii.gz"), force=true)
        cp(joinpath(src_pat_path, "SPECT_DATA", "SPECT_Recon_WholeBody.nii.gz"), joinpath(pat_dir, "SPECT_Recon_WholeBody.nii.gz"), force=true)
        
        println("Saved visualization for $pat_id")
    end
end

# --- Augmentation ---
# Moved to src/utils.jl

function run_training_experiment()
    # Initialize Distributed Backend
    Lux.DistributedUtils.initialize(Lux.MPIBackend)
    backend = Lux.DistributedUtils.get_distributed_backend(Lux.MPIBackend)
    device = Lux.DistributedUtils.get_device(backend)
    if device === nothing device = Lux.gpu_device() end
    rank = Lux.DistributedUtils.local_rank(backend)
    world_size = Lux.DistributedUtils.total_workers(backend)

    rng = Random.default_rng()
    Random.seed!(rng, 1234 + rank)

    if rank == 0
        println("--- Starting Real Data Organ Affine Registration ---")
        println("Distributed Backend: MPI | World Size: $world_size")
    end

    # 1. HDF5 Data Loading
    hdf5_path = "dataset_Lu_50.h5"
    if !isfile(hdf5_path)
        if rank == 0 println("ERROR: HDF5 file not found. Run preprocess_data.jl first.") end
        return
    end

    f = h5open(hdf5_path, "r")
    train_list = read(f["train_list"])
    valid_train = [p for p in train_list if p in keys(f)]
    
    if rank == 0 println("Valid training subjects: $(length(valid_train))") end

    # 2. Model Setup
    num_organs = 4
    model = Chain(
        Conv((3, 3, 3), 2 => 16, stride=2; pad=SamePad()),
        relu,
        Conv((3, 3, 3), 16 => 32, stride=2; pad=SamePad()),
        relu,
        Conv((3, 3, 3), 32 => 64, stride=2; pad=SamePad()),
        relu,
        GlobalMeanPool(),
        FlattenLayer(),
        Dense(64, 128, relu),
        Dense(128, 15 * num_organs)
    )
    
    ps, st = Lux.setup(rng, model)
    ps = ps |> device
    st = st |> device
    ps = Lux.DistributedUtils.synchronize!!(backend, ps)
    st = Lux.DistributedUtils.synchronize!!(backend, st)

    opt = Optimisers.Adam(0.001)
    opt_state = Optimisers.setup(opt, ps)

    # 3. Atlas Points (From one valid patient)
    atlas_points_cpu = fill(-1.0f0, (3, 512, num_organs))
    if !isempty(valid_train)
        first_pat = valid_train[1]
        atlas_vol = read(f[first_pat]["atlas"])
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
    end
    atlas_points = atlas_points_cpu |> device

    # 4. Training Loop
    ka_backend = KernelAbstractions.get_backend(ps.layer_1.weight)
    
    for epoch in 1:20
        shuffle!(rng, valid_train)
        
        for pat_id in valid_train
            # Load Data
            pat_grp = f[pat_id]
            hp_spect = read(pat_grp["spect"])
            hp_atlas = read(pat_grp["atlas"])
            hp_gold = read(pat_grp["gold"])
            
            # 2x Downsample (Strided) to save 8x memory
            spect = hp_spect[1:2:end, 1:2:end, 1:2:end]
            atlas = hp_atlas[1:2:end, 1:2:end, 1:2:end]
            gold = hp_gold[1:2:end, 1:2:end, 1:2:end, :]
            
            x_raw = cat(reshape(atlas, size(atlas)..., 1), 
                         reshape(spect, size(spect)..., 1), dims=4)
            x_raw = reshape(x_raw, size(x_raw)..., 1) |> device
            gold_raw = reshape(gold, size(gold)..., 1) |> device
            
            # Loss Metadata (Loaded from HDF5)
            # Patient spatial centers (half resolution due to 2x strided loading if we use that coords)
            # Actually, metadata in HDF5 is for full resolution. 
            # If we downsample 2x, we must divide coordinates by 2.
            bary_raw = read(pat_grp["barycenters"]) ./ 2.0f0
            radii_raw = read(pat_grp["radii"]) ./ 2.0f0
            
            bary_gpu = bary_raw |> device
            radii_gpu = radii_raw |> device
            
            # Augmentation
            sx, sy, sz = size(atlas)
            centers = Float32[sx/2; sy/2; sz/2]
            centers = reshape(centers, 3, 1)
            
            x_aug, gold_aug, am_meta = Zygote.ignore() do
                apply_random_augmentation(x_raw, gold_raw, centers, ka_backend)
            end

            # Gradient Step
            (l, st), back = Zygote.pullback(p -> begin
                params_pred, new_state = model(x_aug, p, st)
                ap = reshape(params_pred, 15, num_organs, 1)
                l_val = compute_organ_loss(atlas_points, ap, gold_aug[:,:,:,:,1], bary_gpu, radii_gpu)
                return l_val, new_state
            end, ps)
            
            grads = back((1.0f0, nothing))[1]
            grads = allreduce_recursive!!(backend, grads)
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            
            if rank == 0 @printf("Epoch %d | Pat %s | Loss = %.5f\n", epoch, pat_id, l) end
            
            # Memory Management
            x_raw = nothing
            gold_raw = nothing
            x_aug = nothing
            gold_aug = nothing
            GC.gc()
            if ka_backend isa LuxCUDA.CUDABackend
                CUDA.reclaim()
            end
        end
        
        # --- End of Epoch: Visualization ---
        dataset_path = "/home/jm/project_ssd/MedImages.jl/test_data/dataset_Lu"
        visualize_validation(epoch, model, ps, st, valid_train, f, device, ka_backend, rank, dataset_path, num_organs)
    end
    
    close(f)
end

function allreduce_recursive!!(backend, x)
    if x isa NamedTuple
        return NamedTuple{keys(x)}(map(v -> allreduce_recursive!!(backend, v), values(x)))
    elseif x isa Tuple
        return map(v -> allreduce_recursive!!(backend, v), x)
    elseif x isa AbstractArray
        Lux.DistributedUtils.allreduce!(backend, x, Lux.DistributedUtils.avg)
        return x
    else return x end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_training_experiment()
end
