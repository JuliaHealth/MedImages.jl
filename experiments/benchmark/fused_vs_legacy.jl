using MedImages
using MedImages.MedImage_data_struct
using MedImages.Utils
using MedImages.Basic_transformations
using Statistics
using KernelAbstractions

# Legacy Implementation (Mocked for comparison)
function legacy_affine_transform_mi(image::BatchedMedImage, affine_matrices::Union{Matrix{Float64}, Vector{Matrix{Float64}}}, Interpolator::Interpolator_enum; output_size=nothing)
    spatial_size = isnothing(output_size) ? size(image.voxel_data)[1:3] : output_size
    backend = get_backend(image.voxel_data)
    
    # Pre-process matrices like in real affine_transform_mi
    batch_size = size(image.voxel_data, 4)
    matrices_inv_list = map(1:batch_size) do b
        M = (affine_matrices isa Vector) ? affine_matrices[b] : affine_matrices
        M_inv = inv(M)
        return Float32.(M_inv)
    end
    matrices_inv = cat(matrices_inv_list..., dims=3)

    # 1. Generate Coordinates
    points_to_interpolate = generate_affine_coords(spatial_size, matrices_inv, backend)
    
    # 2. Prepare spacing arg
    spacing_arg = [ (1.0, 1.0, 1.0) for _ in 1:batch_size ]
    
    # 3. Perform Interpolation
    resampled_flat = interpolate_my(points_to_interpolate, image.voxel_data, spacing_arg, Interpolator, false, 0.0, true)
    
    return reshape(resampled_flat, spatial_size[1], spatial_size[2], spatial_size[3], batch_size)
end

# Setup
println("--- BENCHMARK: FUSED VS LEGACY AFFINE ---")
size_v = (128, 128, 128)
batch_size = 4
data = rand(Float32, size_v..., batch_size)
imgs = [MedImage(
    voxel_data=data[:,:,:,b], 
    origin=(0.0,0.0,0.0), 
    spacing=(1.0,1.0,1.0), 
    direction=(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0), 
    image_type=MedImage_data_struct.MRI_type, 
    image_subtype=MedImage_data_struct.T1_subtype,
    patient_id="p$b"
) for b in 1:batch_size]
batch = create_batched_medimage(imgs)

mat = Float64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1] # Identity shared

println("Image Size: $size_v, Batch Size: $batch_size")

# Warmup
println("Warming up...")
legacy_affine_transform_mi(batch, mat, Linear_en)
affine_transform_mi(batch, mat, Linear_en)

function time_run(f, args...; n=5)
    t = zeros(n)
    for i in 1:n
        t_start = time_ns()
        f(args...)
        t_end = time_ns()
        t[i] = (t_end - t_start) / 1e9
    end
    return median(t)
end

println("\nRunning Benchmark...")
t_legacy = time_run(legacy_affine_transform_mi, batch, mat, Linear_en)
println("Legacy Implementation Median Time: $(round(t_legacy, digits=4)) s")

t_fused = time_run(affine_transform_mi, batch, mat, Linear_en)
println("Fused Implementation Median Time: $(round(t_fused, digits=4)) s")

println("\nSpeedup: $(round(t_legacy / t_fused, digits=2))x")
