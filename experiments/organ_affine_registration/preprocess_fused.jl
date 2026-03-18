using MedImages
using MedImages.Load_and_save
using MedImages.Resample_to_target
using MedImages.MedImage_data_struct
using MedImages.Basic_transformations
using MedImages.Utils
using HDF5
using ProgressMeter
using Random
using Statistics
using CUDA, LuxCUDA
using KernelAbstractions
using LinearAlgebra
using ChainRulesCore
using Enzyme

# Target Parameters
const TARGET_SPACING = (2.2099, 2.2099, 7.5)
const TARGET_SIZE = (512, 512, 512)
const ATLAS_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]

# Coordinate Grid (cached on GPU)
# Coordinate Grid (cached on GPU)
function get_coords_grid_gpu()
    # 512^3 is too large for a full grid on memory-limited GPUs? 
    # 512^3 * 4 bytes = 536 MB. 3 of them = 1.6 GB. It's okay.
    x = reshape(collect(1.0f0:512.0f0), 512, 1, 1)
    y = reshape(collect(1.0f0:512.0f0), 1, 512, 1)
    z = reshape(collect(1.0f0:512.0f0), 1, 1, 512)
    return CuArray(repeat(x, 1, 512, 512)), CuArray(repeat(y, 512, 1, 512)), CuArray(repeat(z, 512, 512, 1))
end

function get_fused_preprocess_matrix(source_mi, target_size, target_spacing)
    # Target orientation is RAS
    # Direction matrix for RAS as per Orientation_dicts
    # MedImage_data_struct.ORIENTATION_RAS => (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    D_tgt = [-1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 1.0]
    S_tgt = Diagonal(collect(target_spacing))
    M_idx2phys_tgt = D_tgt * S_tgt
    
    # Source metadata
    D_src = reshape(collect(source_mi.direction), 3, 3)
    S_src = Diagonal(collect(source_mi.spacing))
    M_idx2phys_src = D_src * S_src
    M_phys2idx_src = inv(M_idx2phys_src)
    
    # Calculate target origin to center source volume in target grid
    src_size = size(source_mi.voxel_data)
    # Physical center of source
    src_center_idx = (collect(src_size) .- 1.0) ./ 2.0
    src_center_phys = collect(source_mi.origin) .+ M_idx2phys_src * src_center_idx
    
    # Target center in index space
    tgt_center_idx = (collect(target_size) .- 1.0) ./ 2.0
    # We want M_idx2phys_tgt * tgt_center_idx + origin_tgt = src_center_phys
    origin_tgt = src_center_phys .- M_idx2phys_tgt * tgt_center_idx
    
    # Combined transform: index_src = M_phys2idx_src * (M_idx2phys_tgt * index_tgt + origin_tgt - origin_src)
    # index_src = (M_phys2idx_src * M_idx2phys_tgt) * index_tgt + M_phys2idx_src * (origin_tgt - origin_src)
    
    M_combined = Matrix{Float64}(I, 4, 4)
    M_combined[1:3, 1:3] = M_phys2idx_src * M_idx2phys_tgt
    M_combined[1:3, 4] = M_phys2idx_src * (origin_tgt .- collect(source_mi.origin))
    
    # Adjust for 1-based indexing in kernel
    # In kernel: ix, iy, iz are 1..512
    # So idx_0based = idx_1based - 1
    # index_src_0based = M * (index_tgt_1based - 1) + offset
    # index_src_1based = index_src_0based + 1 = M * index_tgt_1based - M * 1 + offset + 1
    
    M_final = Matrix{Float64}(I, 4, 4)
    M_final[1:3, 1:3] = M_combined[1:3, 1:3]
    M_final[1:3, 4] = M_combined[1:3, 4] .- M_combined[1:3, 1:3] * ones(3) .+ ones(3)
    
    return M_final
end

# --- Fused Affine Kernel (Enzyme-compatible) ---

@kernel function fused_affine_kernel!(out, src, M, target_size_arr, src_dims_arr, is_nearest)
    i = @index(Global, Linear)
    
    # 1. Map linear index to target (x,y,z)
    tx_size = Int(target_size_arr[1])
    ty_size = Int(target_size_arr[2])
    tz_size = Int(target_size_arr[3])
    
    stride_z = tx_size * ty_size
    iz = (i - 1) ÷ stride_z + 1
    rem_z = (i - 1) % stride_z
    iy = rem_z ÷ tx_size + 1
    ix = rem_z % tx_size + 1
    
    # 2. Apply Affine Transform
    # M is (4, 4)
    px = M[1,1]*ix + M[1,2]*iy + M[1,3]*iz + M[1,4]
    py = M[2,1]*ix + M[2,2]*iy + M[2,3]*iz + M[2,4]
    pz = M[3,1]*ix + M[3,2]*iy + M[3,3]*iz + M[3,4]
    
    # 3. Interpolate
    sx = Int(src_dims_arr[1])
    sy = Int(src_dims_arr[2])
    sz = Int(src_dims_arr[3])
    
    if px < 1.0f0 || py < 1.0f0 || pz < 1.0f0 || px > Float32(sx) || py > Float32(sy) || pz > Float32(sz)
        out[i] = 0.0f0
    else
        if is_nearest
            nx = Int(round(px))
            ny = Int(round(py))
            nz = Int(round(pz))
            # Clamp to be safe
            nx = max(1, min(nx, sx))
            ny = max(1, min(ny, sy))
            nz = max(1, min(nz, sz))
            @inbounds out[i] = Float32(src[nx, ny, nz])
        else
            # Trilinear (Unrolled)
            x0 = floor(Int, px)
            y0 = floor(Int, py)
            z0 = floor(Int, pz)
            
            x1 = min(x0 + 1, sx)
            y1 = min(y0 + 1, sy)
            z1 = min(z0 + 1, sz)
            
            xd = px - x0
            yd = py - y0
            zd = pz - z0
            
            @inbounds begin
                # Unrolled trilinear interpolation
                v000 = Float32(src[x0, y0, z0])
                v100 = Float32(src[x1, y0, z0])
                v010 = Float32(src[x0, y1, z0])
                v110 = Float32(src[x1, y1, z0])
                v001 = Float32(src[x0, y0, z1])
                v101 = Float32(src[x1, y0, z1])
                v011 = Float32(src[x0, y1, z1])
                v111 = Float32(src[x1, y1, z1])

                c00 = v000 * (1.0f0 - xd) + v100 * xd
                c10 = v010 * (1.0f0 - xd) + v110 * xd
                c01 = v001 * (1.0f0 - xd) + v101 * xd
                c11 = v011 * (1.0f0 - xd) + v111 * xd

                c0 = c00 * (1.0f0 - yd) + c10 * yd
                c1 = c01 * (1.0f0 - yd) + c11 * yd

                out[i] = c0 * (1.0f0 - zd) + c1 * zd
            end
        end
    end
end

function interpolate_fused_affine_launcher!(out, src, M, target_size_arr, src_dims_arr, is_nearest, ndrange_val)
    backend = KernelAbstractions.get_backend(out)
    kernel = fused_affine_kernel!(backend, 256)
    kernel(out, src, M, target_size_arr, src_dims_arr, is_nearest, ndrange=ndrange_val)
    KernelAbstractions.synchronize(backend)
    return nothing
end

function interpolate_fused_affine(src, M, target_size, interp_mode)
    backend = KernelAbstractions.get_backend(src)
    out = KernelAbstractions.zeros(backend, Float32, prod(target_size))
    
    # Metadata for kernel
    src_dims_arr = CuArray(Int32[size(src)...])
    target_size_arr = CuArray(Int32[target_size...])
    is_nearest = (interp_mode == Nearest_neighbour_en)
    
    ndrange_val = prod(target_size)
    interpolate_fused_affine_launcher!(out, src, M, target_size_arr, src_dims_arr, is_nearest, ndrange_val)
    
    return reshape(out, target_size)
end

function ChainRulesCore.rrule(::typeof(interpolate_fused_affine), src, M, target_size, interp_mode)
    output = interpolate_fused_affine(src, M, target_size, interp_mode)
    
    function interpolate_pullback(d_output_unthunked)
        d_output = unthunk(d_output_unthunked)
        backend = KernelAbstractions.get_backend(src)
        
        # Gradients
        d_src = KernelAbstractions.zeros(backend, Float32, size(src))
        d_M = KernelAbstractions.zeros(backend, Float32, size(M))
        
        # Metadata
        src_dims_arr = CuArray(Int32[size(src)...])
        target_size_arr = CuArray(Int32[target_size...])
        is_nearest = (interp_mode == Nearest_neighbour_en)
        ndrange_val = prod(target_size)
        
        # Shadow arrays for metadata (Enzyme often likes Duplicated for all arguments if not strictly Const)
        d_t_dims = CUDA.zeros(Int32, 3)
        d_s_dims = CUDA.zeros(Int32, 3)
        
        Enzyme.autodiff(
            Reverse,
            interpolate_fused_affine_launcher!,
            Const,
            Duplicated(vec(output), vec(d_output)),
            Duplicated(src, d_src),
            Duplicated(M, d_M),
            Duplicated(target_size_arr, d_t_dims),
            Duplicated(src_dims_arr, d_s_dims),
            Const(is_nearest),
            Const(ndrange_val)
        )
        
        return NoTangent(), d_src, d_M, NoTangent(), NoTangent()
    end
    
    return output, interpolate_pullback
end

# --- Preprocessing Logic ---

function process_subject_differentiable(id, img_path, seg_path, labels_gpu, grid_gpu; store_mode=:none, h5_file=nothing)
    try
        # 1. Load Data
        img_mi = load_image(img_path, "PET")
        seg_mi = load_image(seg_path, "CT")
        
        # 2. Fused Affine Matrix
        M = get_fused_preprocess_matrix(img_mi, TARGET_SIZE, TARGET_SPACING)
        M_gpu = CuArray(Float32.(M))
        
        # 3. GPU Interpolation
        img_gpu = CuArray(Float32.(img_mi.voxel_data))
        seg_gpu = CuArray(Int32.(seg_mi.voxel_data))
        
        interp_time = CUDA.@elapsed begin
            img_final = interpolate_fused_affine(img_gpu, M_gpu, TARGET_SIZE, Linear_en)
            seg_final = interpolate_fused_affine(seg_gpu, M_gpu, TARGET_SIZE, Nearest_neighbour_en)
        end
        
        # 4. Differentiable Stats
        n_labels = length(ATLAS_LABELS)
        barycenters = zeros(Float32, 3, n_labels)
        radii_out = zeros(Float32, 1, n_labels)
        
        gx, gy, gz = grid_gpu
        
        for i in 1:n_labels
            lbl = ATLAS_LABELS[i]
            # Process one label at a time to save memory (512^3 is huge)
            gold_onehot = Float32.(seg_final .== lbl)
            counts = sum(gold_onehot) + 1f-8
            
            b_x = sum(gold_onehot .* gx) / counts
            b_y = sum(gold_onehot .* gy) / counts
            b_z = sum(gold_onehot .* gz) / counts
            
            barycenters[1, i] = b_x
            barycenters[2, i] = b_y
            barycenters[3, i] = b_z
            
            dist_sq = ((gx .- b_x).^2 .+ (gy .- b_y).^2 .+ (gz .- b_z).^2) .* gold_onehot
            r = sqrt(maximum(dist_sq))
            radii_out[1, i] = max(r, 1.0f0)
        end
        
        # 5. Conditional Storage
        if store_mode != :none && h5_file !== nothing
            g = create_group(h5_file, id)
            g["barycenters"] = barycenters
            g["radii"] = radii_out
            
            if store_mode == :full
                g["image"] = Array(img_final)
                # Omit saving full 58GB gold tensor, maybe save just seg_final
                g["segmentation"] = Array(seg_final)
            end
        end
        
        # Cleanup
        CUDA.reclaim()
        GC.gc()
        return true, interp_time
    catch e
        println("ERROR [$(id)]: $e")
        Base.display_error(e, catch_backtrace())
        return false, 0.0
    end
end

function run_optimized_benchmark(n_subjects=10; store_mode=:none, output_hdf5="dataset_unified_fused_10.h5")
    subjects = []
    dataset_path = "/home/jm/project_ssd/MedImages.jl/test_data/dataset_PET/"
    find_cmd = `find $dataset_path -name "*-nukl.nii.gz"`
    all_pet_files = readlines(pipeline(find_cmd))
    
    n_added = 0
    for pet_path in all_pet_files
        dir_path = dirname(pet_path)
        base_name = basename(pet_path)
        id = replace(base_name, "-nukl.nii.gz" => "")
        ct_path = joinpath(dir_path, "$id.ct.nii.gz")
        seg_path = joinpath(dir_path, "seg_ct.nii.gz")
        
        if isfile(ct_path) && isfile(seg_path)
            push!(subjects, (id, pet_path, seg_path))
            n_added += 1
            if n_added == n_subjects
                break
            end
        end
    end
    
    n_run = min(n_subjects, length(subjects))
    println("--- Raw GPU Throughput Benchmark ($n_run subjects, Mode: $store_mode) ---")
    
    labels_gpu = CuArray(Int32.(reshape(ATLAS_LABELS, 1, 1, 1, :)))
    gx, gy, gz = get_coords_grid_gpu()
    grid_gpu = (gx, gy, gz)
    
    f = (store_mode != :none) ? h5open(output_hdf5, "w") : nothing
    
    start_time = time()
    total_interp_time = 0.0
    @showprogress for i in 1:n_run
        id, img, seg = subjects[i]
        success, interp_t = process_subject_differentiable(id, img, seg, labels_gpu, grid_gpu, store_mode=store_mode, h5_file=f)
        if success
            total_interp_time += interp_t
        end
    end
    end_time = time()
    
    total_time = end_time - start_time
    println("\nResults:")
    println("Total End-to-End Time: $(round(total_time, digits=2))s")
    println("Average End-to-End: $(round(total_time/n_run, digits=4))s/subject")
    
    println("\n--- Fused Kernel GPU Interpolation Only ---")
    println("Total GPU Kernel Time: $(round(total_interp_time, digits=2))s")
    println("Average GPU Kernel Time: $(round(total_interp_time/n_run, digits=4))s/subject")
    
    if f !== nothing; close(f) end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Run 100 subjects as requested
    run_optimized_benchmark(100, store_mode=:none)
end
