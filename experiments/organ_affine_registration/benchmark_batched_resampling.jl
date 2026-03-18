using MedImages
using MedImages.Load_and_save
using MedImages.Resample_to_target
using MedImages.MedImage_data_struct
using MedImages.Basic_transformations
using MedImages.Utils
using ProgressMeter
using KernelAbstractions
using CUDA
using LinearAlgebra

# Target Grid
const TARGET_SIZE = (512, 512, 512) # Same as single-subject test
const TARGET_SPACING = (2.2099, 2.2099, 7.5)

@kernel function batched_fused_affine_kernel!(out, src, M_batch, target_size_arr, src_dims_arr, is_nearest)
    i = @index(Global, Linear)
    
    # 1. Map linear index to target (x,y,z,b)
    tx = Int(target_size_arr[1])
    ty = Int(target_size_arr[2])
    tz = Int(target_size_arr[3])
    
    stride_z = tx * ty
    stride_b = stride_z * tz
    
    # Batch index
    ib = (i - 1) ÷ stride_b + 1
    rem_b = (i - 1) % stride_b
    
    # spatial indices
    iz = rem_b ÷ stride_z + 1
    rem_z = rem_b % stride_z
    iy = rem_z ÷ tx + 1
    ix = rem_z % tx + 1
    
    # 2. Apply Affine Transform
    # M_batch is (4, 4, B)
    m11 = M_batch[1, 1, ib]
    m12 = M_batch[1, 2, ib]
    m13 = M_batch[1, 3, ib]
    m14 = M_batch[1, 4, ib]
    
    m21 = M_batch[2, 1, ib]
    m22 = M_batch[2, 2, ib]
    m23 = M_batch[2, 3, ib]
    m24 = M_batch[2, 4, ib]
    
    m31 = M_batch[3, 1, ib]
    m32 = M_batch[3, 2, ib]
    m33 = M_batch[3, 3, ib]
    m34 = M_batch[3, 4, ib]
    
    px = m11*ix + m12*iy + m13*iz + m14
    py = m21*ix + m22*iy + m23*iz + m24
    pz = m31*ix + m32*iy + m33*iz + m34
    
    # 3. Interpolate from src (x,y,z,b)
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
            nx = max(1, min(nx, sx))
            ny = max(1, min(ny, sy))
            nz = max(1, min(nz, sz))
            @inbounds out[i] = Float32(src[nx, ny, nz, ib])
        else
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
                v000 = Float32(src[x0, y0, z0, ib])
                v100 = Float32(src[x1, y0, z0, ib])
                v010 = Float32(src[x0, y1, z0, ib])
                v110 = Float32(src[x1, y1, z0, ib])
                v001 = Float32(src[x0, y0, z1, ib])
                v101 = Float32(src[x1, y0, z1, ib])
                v011 = Float32(src[x0, y1, z1, ib])
                v111 = Float32(src[x1, y1, z1, ib])

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

function get_fused_preprocess_matrix(source_mi, target_size, target_spacing)
    D_tgt = [-1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 1.0]
    S_tgt = Diagonal(collect(target_spacing))
    M_idx2phys_tgt = D_tgt * S_tgt
    
    D_src = reshape(collect(source_mi.direction), 3, 3)
    S_src = Diagonal(collect(source_mi.spacing))
    M_idx2phys_src = D_src * S_src
    M_phys2idx_src = inv(M_idx2phys_src)
    
    src_size = size(source_mi.voxel_data)
    src_center_idx = (collect(src_size) .- 1.0) ./ 2.0
    src_center_phys = collect(source_mi.origin) .+ M_idx2phys_src * src_center_idx
    
    tgt_center_idx = (collect(target_size) .- 1.0) ./ 2.0
    origin_tgt = src_center_phys .- M_idx2phys_tgt * tgt_center_idx
    
    M_combined = Matrix{Float64}(I, 4, 4)
    M_combined[1:3, 1:3] = M_phys2idx_src * M_idx2phys_tgt
    M_combined[1:3, 4] = M_phys2idx_src * (origin_tgt .- collect(source_mi.origin))
    
    M_final = Matrix{Float64}(I, 4, 4)
    M_final[1:3, 1:3] = M_combined[1:3, 1:3]
    M_final[1:3, 4] = M_combined[1:3, 4] .- M_combined[1:3, 1:3] * ones(3) .+ ones(3)
    
    return M_final
end

function benchmark_batched_resampling(total_subjects=100, batch_size=10)
    println("--- Batched PET-to-CT Resampling Benchmark (Total: $total_subjects, Batch Size: $batch_size) ---")
    
    dataset_path = "/home/jm/project_ssd/MedImages.jl/test_data/dataset_PET/"
    find_cmd = `find $dataset_path -name "*-nukl.nii.gz"`
    all_pet_files = readlines(pipeline(find_cmd))
    
    subjects_mi = []
    
    for pet_path in all_pet_files
        dir_path = dirname(pet_path)
        base_name = basename(pet_path)
        id = replace(base_name, "-nukl.nii.gz" => "")
        ct_path = joinpath(dir_path, "$id.ct.nii.gz")
        
        if isfile(ct_path)
            pet_mi = load_image(pet_path, "PET")
            ct_mi  = load_image(ct_path, "CT")
            push!(subjects_mi, (pet_mi, ct_mi))
            if length(subjects_mi) == total_subjects
                break
            end
        end
    end
    
    if length(subjects_mi) < 1
        println("Warning: Found no valid pairs.")
        return
    end
    
    actual_total = length(subjects_mi)
    pet_batch_voxel = zeros(Float32, size(subjects_mi[1][1].voxel_data)..., batch_size)
    M_batch = zeros(Float32, 4, 4, batch_size)
    
    pet_gpu = CUDA.zeros(Float32, size(pet_batch_voxel))
    M_gpu = CUDA.zeros(Float32, size(M_batch))
    
    backend = KernelAbstractions.get_backend(pet_gpu)
    out_gpu = KernelAbstractions.zeros(backend, Float32, TARGET_SIZE[1]*TARGET_SIZE[2]*TARGET_SIZE[3]*batch_size)
    
    src_dims_arr = CuArray(Int32[size(subjects_mi[1][1].voxel_data)...])
    target_size_arr = CuArray(Int32[TARGET_SIZE...])
    ndrange_val = length(out_gpu)
    kernel = batched_fused_affine_kernel!(backend, 256)
    
    total_interp_time = 0.0
    num_batches = ceil(Int, actual_total / batch_size)
    
    start_time = time()
    for batch_idx in 1:num_batches
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, actual_total)
        current_batch_size = end_idx - start_idx + 1
        
        for b in 1:current_batch_size
            subj_idx = start_idx + b - 1
            # To avoid dimension mismatch from variable clinical sizes, 
            # we use the first subject's dimensions for pure compute throughput benching
            pet_batch_voxel[:, :, :, b] .= subjects_mi[1][1].voxel_data
            M_batch[:, :, b] = get_fused_preprocess_matrix(subjects_mi[1][1], TARGET_SIZE, TARGET_SPACING)
        end
        
        copyto!(pet_gpu, pet_batch_voxel)
        copyto!(M_gpu, M_batch)
        
        # Use full GPU array but adjust ndrange for only the valid items if partial batch?
        # For benchmark throughput, we can just process the full padded memory if current_batch_size < batch_size.
        
        interp_time = CUDA.@elapsed begin
            kernel(out_gpu, pet_gpu, M_gpu, target_size_arr, src_dims_arr, false, ndrange=ndrange_val)
            KernelAbstractions.synchronize(backend)
        end
        total_interp_time += interp_time
    end
    end_time = time()
    
    wall_clock = end_time - start_time
    println("\nResults:")
    println("Total End-to-End Time ($actual_total subjects): $(round(wall_clock, digits=2))s")
    println("Average End-to-End: $(round(wall_clock/actual_total, digits=4))s/subject")
    
    println("\n--- Fused Kernel GPU Interpolation Only ---")
    println("Total Batched GPU Kernel Time ($actual_total subjects): $(round(total_interp_time, digits=4))s")
    println("Average GPU Kernel Time: $(round(total_interp_time/actual_total, digits=4))s/subject")
    
    CUDA.reclaim()
    GC.gc()
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_batched_resampling(100, 10)
end
