"""
GPU Benchmarks
Core benchmarking functions for MedImages.jl GPU operations.
"""

# Note: MedImages should already be loaded by run_gpu_benchmarks.jl
# which handles project activation

using MedImages
using BenchmarkTools
using Statistics
using Printf
using Dates

# Import CUDA if available
CUDA_AVAILABLE = false
try
    using CUDA
    global CUDA_AVAILABLE = CUDA.functional()
    if CUDA_AVAILABLE
        dev = CUDA.device()
        println("CUDA available: $(CUDA.name(dev))")
        println("CUDA memory: $(CUDA.totalmem(dev) ÷ 1024^3) GB")
    end
catch e
    @warn "CUDA.jl not available or non-functional" exception=e
end

# benchmark_config.jl is included by run_gpu_benchmarks.jl

"""
Benchmark result structure
"""
struct BenchmarkResult
    name::String
    operation::String
    backend::String
    image_size::String
    parameters::Dict{String, Any}
    time_mean::Float64     # seconds
    time_median::Float64
    time_std::Float64
    memory_bytes::Int64
    throughput::Float64    # operation-specific (voxels/sec, points/sec, etc.)
    timestamp::DateTime
end

"""
    warmup_gpu(image::MedImage, iterations::Int=3)

Warm up GPU with dummy operations to ensure accurate benchmarking.
"""
function warmup_gpu(image::MedImage, iterations::Int=WARMUP_ITERATIONS)
    if !CUDA_AVAILABLE
        return
    end

    println("Warming up GPU...")

    for i in 1:iterations
        # Perform a dummy resampling operation
        try
            result = MedImages.resample_to_spacing(
                image,
                (image.spacing[1] * 1.1, image.spacing[2] * 1.1, image.spacing[3] * 1.1),
                MedImages.Linear_en
            )
            CUDA.@sync nothing  # Ensure completion
        catch e
            @warn "Warmup iteration failed" iteration=i exception=e
        end
    end

    println("  GPU warmed up")
end

"""
    transfer_to_gpu(image::MedImage) -> MedImage

Transfer image data to GPU memory.
"""
function transfer_to_gpu(image::MedImage)
    if !CUDA_AVAILABLE
        return image
    end

    # Create copy with GPU array
    gpu_image = deepcopy(image)
    gpu_image.voxel_data = CuArray(Float32.(gpu_image.voxel_data))
    gpu_image.current_device = MedImages.MedImage_data_struct.CUDA_current_device

    return gpu_image
end

"""
    benchmark_interpolate_kernel(image::MedImage, point_count::Int, interpolator::MedImages.Interpolator_enum, backend::String) -> BenchmarkResult

Benchmark the core interpolation kernel.
"""
function benchmark_interpolate_kernel(image::MedImage, point_count::Int, interpolator::MedImages.Interpolator_enum, backend::String)
    @printf("\nBenchmarking interpolate_kernel: %s backend, %d points, %s interpolation\n",
            backend, point_count, interpolator == MedImages.Nearest_neighbour_en ? "Nearest" : "Linear")

    dims = size(image.voxel_data)

    # Generate random interpolation points within image bounds
    # Scale factors for each dimension (as column vector for proper broadcasting)
    scale = Float32.([dims[1]-1, dims[2]-1, dims[3]-1])
    points = rand(Float32, 3, point_count) .* scale .+ 1.0f0

    # Transfer to GPU if needed
    test_image = backend == "CUDA" ? transfer_to_gpu(image) : image
    test_points = backend == "CUDA" ? CuArray(points) : points

    # Warmup
    if backend == "CUDA"
        warmup_gpu(test_image)
    end

    # Benchmark
    println("  Running benchmark...")
    b = @benchmark begin
        result = MedImages.interpolate_my(
            $test_points,
            $(test_image.voxel_data),
            $(test_image.spacing),
            $interpolator,
            false,  # keep_beginning_same
            0,      # extrapolate_value
            true    # use_fast (GPU kernel)
        )
        if $backend == "CUDA"
            CUDA.@sync nothing
        end
    end samples=BENCHMARK_SAMPLES seconds=BENCHMARK_SECONDS

    # Calculate throughput (points per second)
    throughput = point_count / (median(b).time * 1e-9)

    @printf("  Time: %.3f ms (median), %.3f ms (mean), %.3f ms (std)\n",
            median(b).time * 1e-6, mean(b).time * 1e-6, std(b).time * 1e-6)
    @printf("  Throughput: %.2e points/sec\n", throughput)
    @printf("  Memory: %.1f MB\n", b.memory / 1024^2)

    return BenchmarkResult(
        "interpolate_kernel",
        "interpolation",
        backend,
        "$(dims[1])x$(dims[2])x$(dims[3])",
        Dict("point_count" => point_count, "interpolator" => string(interpolator)),
        mean(b).time * 1e-9,
        median(b).time * 1e-9,
        std(b).time * 1e-9,
        b.memory,
        throughput,
        now()
    )
end

"""
    benchmark_resample_to_spacing(image::MedImage, factor::Float64, interpolator::MedImages.Interpolator_enum, backend::String) -> BenchmarkResult

Benchmark resampling to different spacing.
"""
function benchmark_resample_to_spacing(image::MedImage, factor::Float64, interpolator::MedImages.Interpolator_enum, backend::String)
    factor_name = factor > 1.0 ? "upsample" : "downsample"
    @printf("\nBenchmarking resample_to_spacing: %s backend, %s %.2fx, %s interpolation\n",
            backend, factor_name, factor, interpolator == MedImages.Nearest_neighbour_en ? "Nearest" : "Linear")

    dims = size(image.voxel_data)
    new_spacing = (image.spacing[1] / factor, image.spacing[2] / factor, image.spacing[3] / factor)

    # Transfer to GPU if needed
    test_image = backend == "CUDA" ? transfer_to_gpu(image) : image

    # Warmup
    if backend == "CUDA"
        warmup_gpu(test_image)
    end

    # Run once to get output dimensions
    result_sample = MedImages.resample_to_spacing(test_image, new_spacing, interpolator)
    new_dims = size(result_sample.voxel_data)

    # Benchmark
    println("  Running benchmark...")
    b = @benchmark begin
        result = MedImages.resample_to_spacing($test_image, $new_spacing, $interpolator)
        if $backend == "CUDA"
            CUDA.@sync nothing
        end
    end samples=BENCHMARK_SAMPLES seconds=BENCHMARK_SECONDS

    # Calculate throughput (input voxels per second)
    voxel_count = prod(dims)
    throughput = voxel_count / (median(b).time * 1e-9)

    @printf("  Input dims: %dx%dx%d\n", dims...)
    @printf("  Output dims: %dx%dx%d\n", new_dims...)
    @printf("  Time: %.3f ms (median), %.3f ms (mean), %.3f ms (std)\n",
            median(b).time * 1e-6, mean(b).time * 1e-6, std(b).time * 1e-6)
    @printf("  Throughput: %.2e voxels/sec\n", throughput)
    @printf("  Memory: %.1f MB\n", b.memory / 1024^2)

    return BenchmarkResult(
        "resample_to_spacing",
        "resampling",
        backend,
        "$(dims[1])x$(dims[2])x$(dims[3])",
        Dict("factor" => factor, "interpolator" => string(interpolator), "new_spacing" => new_spacing),
        mean(b).time * 1e-9,
        median(b).time * 1e-9,
        std(b).time * 1e-9,
        b.memory,
        throughput,
        now()
    )
end

"""
    benchmark_resample_to_image(source_image::MedImage, target_image::MedImage, interpolator::MedImages.Interpolator_enum, backend::String) -> BenchmarkResult

Benchmark cross-image resampling (includes orientation changes).
"""
function benchmark_resample_to_image(source_image::MedImage, target_image::MedImage, interpolator::MedImages.Interpolator_enum, backend::String)
    @printf("\nBenchmarking resample_to_image: %s backend, %s interpolation\n",
            backend, interpolator == MedImages.Nearest_neighbour_en ? "Nearest" : "Linear")

    dims = size(source_image.voxel_data)

    # Transfer to GPU if needed
    test_source = backend == "CUDA" ? transfer_to_gpu(source_image) : source_image
    test_target = backend == "CUDA" ? transfer_to_gpu(target_image) : target_image

    # Warmup
    if backend == "CUDA"
        warmup_gpu(test_source)
    end

    # Benchmark
    println("  Running benchmark...")
    b = @benchmark begin
        result = MedImages.resample_to_image($test_source, $test_target, $interpolator)
        if $backend == "CUDA"
            CUDA.@sync nothing
        end
    end samples=BENCHMARK_SAMPLES seconds=BENCHMARK_SECONDS

    # Calculate throughput
    voxel_count = prod(dims)
    throughput = voxel_count / (median(b).time * 1e-9)

    @printf("  Time: %.3f ms (median), %.3f ms (mean), %.3f ms (std)\n",
            median(b).time * 1e-6, mean(b).time * 1e-6, std(b).time * 1e-6)
    @printf("  Throughput: %.2e voxels/sec\n", throughput)
    @printf("  Memory: %.1f MB\n", b.memory / 1024^2)

    return BenchmarkResult(
        "resample_to_image",
        "cross_image_resampling",
        backend,
        "$(dims[1])x$(dims[2])x$(dims[3])",
        Dict("interpolator" => string(interpolator)),
        mean(b).time * 1e-9,
        median(b).time * 1e-9,
        std(b).time * 1e-9,
        b.memory,
        throughput,
        now()
    )
end

"""
    benchmark_rotation(image::MedImage, angle::Int, backend::String) -> BenchmarkResult

Benchmark rotation operation.
"""
function benchmark_rotation(image::MedImage, angle::Int, backend::String)
    @printf("\nBenchmarking rotate_mi: %s backend, %d degrees\n", backend, angle)

    dims = size(image.voxel_data)

    # Transfer to GPU if needed
    test_image = backend == "CUDA" ? transfer_to_gpu(image) : image

    # Determine axis based on angle (rotate around z-axis)
    axis = 3

    # Warmup
    if backend == "CUDA"
        warmup_gpu(test_image)
    end

    # Benchmark
    println("  Running benchmark...")
    b = @benchmark begin
        result = MedImages.rotate_mi($test_image, $axis, Float64($angle), MedImages.Linear_en)
        if $backend == "CUDA"
            CUDA.@sync nothing
        end
    end samples=BENCHMARK_SAMPLES seconds=BENCHMARK_SECONDS

    # Calculate throughput
    voxel_count = prod(dims)
    throughput = voxel_count / (median(b).time * 1e-9)

    @printf("  Time: %.3f ms (median), %.3f ms (mean), %.3f ms (std)\n",
            median(b).time * 1e-6, mean(b).time * 1e-6, std(b).time * 1e-6)
    @printf("  Throughput: %.2e voxels/sec\n", throughput)
    @printf("  Memory: %.1f MB\n", b.memory / 1024^2)

    return BenchmarkResult(
        "rotate_mi",
        "rotation",
        backend,
        "$(dims[1])x$(dims[2])x$(dims[3])",
        Dict("angle" => angle, "axis" => axis),
        mean(b).time * 1e-9,
        median(b).time * 1e-9,
        std(b).time * 1e-9,
        b.memory,
        throughput,
        now()
    )
end

"""
    benchmark_crop(image::MedImage, ratio::Float64, backend::String) -> BenchmarkResult

Benchmark crop operation.
"""
function benchmark_crop(image::MedImage, ratio::Float64, backend::String)
    @printf("\nBenchmarking crop_mi: %s backend, %.1f%% crop\n", backend, ratio * 100)

    dims = size(image.voxel_data)

    # Calculate crop: crop_beg is start position, crop_size is remaining dimensions
    # Crop 'ratio' proportion from each side
    crop_offset = Tuple(Int(round(d * ratio / 2)) for d in dims)
    crop_size = Tuple(dims[i] - 2 * crop_offset[i] for i in 1:3)

    # Transfer to GPU if needed
    test_image = backend == "CUDA" ? transfer_to_gpu(image) : image

    # Benchmark
    println("  Running benchmark...")
    b = @benchmark begin
        result = MedImages.crop_mi($test_image, $crop_offset, $crop_size, MedImages.Linear_en)
        if $backend == "CUDA"
            CUDA.@sync nothing
        end
    end samples=BENCHMARK_SAMPLES seconds=BENCHMARK_SECONDS

    # Calculate throughput
    voxel_count = prod(dims)
    throughput = voxel_count / (median(b).time * 1e-9)

    @printf("  Time: %.3f ms (median), %.3f ms (mean), %.3f ms (std)\n",
            median(b).time * 1e-6, mean(b).time * 1e-6, std(b).time * 1e-6)
    @printf("  Throughput: %.2e voxels/sec\n", throughput)
    @printf("  Memory: %.1f MB\n", b.memory / 1024^2)

    return BenchmarkResult(
        "crop_mi",
        "crop",
        backend,
        "$(dims[1])x$(dims[2])x$(dims[3])",
        Dict("ratio" => ratio, "crop_offset" => crop_offset, "crop_size" => crop_size),
        mean(b).time * 1e-9,
        median(b).time * 1e-9,
        std(b).time * 1e-9,
        b.memory,
        throughput,
        now()
    )
end

"""
    benchmark_pad(image::MedImage, ratio::Float64, backend::String) -> BenchmarkResult

Benchmark pad operation.
"""
function benchmark_pad(image::MedImage, ratio::Float64, backend::String)
    @printf("\nBenchmarking pad_mi: %s backend, %.1f%% pad\n", backend, ratio * 100)

    dims = size(image.voxel_data)

    # Calculate pad amount as Tuple
    pad_amount = Tuple(Int(round(d * ratio / 2)) for d in dims)
    pad_val = 0.0f0  # Pad with zeros

    # Transfer to GPU if needed
    test_image = backend == "CUDA" ? transfer_to_gpu(image) : image

    # Benchmark
    println("  Running benchmark...")
    b = @benchmark begin
        result = MedImages.pad_mi($test_image, $pad_amount, $pad_amount, $pad_val, MedImages.Linear_en)
        if $backend == "CUDA"
            CUDA.@sync nothing
        end
    end samples=BENCHMARK_SAMPLES seconds=BENCHMARK_SECONDS

    # Calculate throughput
    voxel_count = prod(dims)
    throughput = voxel_count / (median(b).time * 1e-9)

    @printf("  Time: %.3f ms (median), %.3f ms (mean), %.3f ms (std)\n",
            median(b).time * 1e-6, mean(b).time * 1e-6, std(b).time * 1e-6)
    @printf("  Throughput: %.2e voxels/sec\n", throughput)
    @printf("  Memory: %.1f MB\n", b.memory / 1024^2)

    return BenchmarkResult(
        "pad_mi",
        "pad",
        backend,
        "$(dims[1])x$(dims[2])x$(dims[3])",
        Dict("ratio" => ratio, "pad_amount" => pad_amount),
        mean(b).time * 1e-9,
        median(b).time * 1e-9,
        std(b).time * 1e-9,
        b.memory,
        throughput,
        now()
    )
end

"""
    benchmark_change_orientation(image::MedImage, target_orientation::MedImages.Orientation_code, backend::String) -> BenchmarkResult

Benchmark orientation change operation.
"""
function benchmark_change_orientation(image::MedImage, target_orientation::MedImages.Orientation_code, backend::String)
    @printf("\nBenchmarking change_orientation: %s backend, target: %s\n", backend, target_orientation)

    dims = size(image.voxel_data)

    # Transfer to GPU if needed
    test_image = backend == "CUDA" ? transfer_to_gpu(image) : image

    # Benchmark
    println("  Running benchmark...")
    b = @benchmark begin
        result = MedImages.change_orientation($test_image, $target_orientation)
        if $backend == "CUDA"
            CUDA.@sync nothing
        end
    end samples=BENCHMARK_SAMPLES seconds=BENCHMARK_SECONDS

    # Calculate throughput
    voxel_count = prod(dims)
    throughput = voxel_count / (median(b).time * 1e-9)

    @printf("  Time: %.3f ms (median), %.3f ms (mean), %.3f ms (std)\n",
            median(b).time * 1e-6, mean(b).time * 1e-6, std(b).time * 1e-6)
    @printf("  Throughput: %.2e voxels/sec\n", throughput)
    @printf("  Memory: %.1f MB\n", b.memory / 1024^2)

    return BenchmarkResult(
        "change_orientation",
        "orientation",
        backend,
        "$(dims[1])x$(dims[2])x$(dims[3])",
        Dict("target_orientation" => string(target_orientation)),
        mean(b).time * 1e-9,
        median(b).time * 1e-9,
        std(b).time * 1e-9,
        b.memory,
        throughput,
        now()
    )
end

"""
    benchmark_memory_transfer(image::MedImage) -> Dict

Benchmark CPU <-> GPU memory transfer performance.
"""
function benchmark_memory_transfer(image::MedImage)
    if !CUDA_AVAILABLE
        @warn "CUDA not available, skipping memory transfer benchmark"
        return Dict()
    end

    @printf("\nBenchmarking memory transfer: %s\n", size(image.voxel_data))

    dims = size(image.voxel_data)
    data_size = prod(dims) * sizeof(eltype(image.voxel_data)) / 1024^2

    results = Dict()

    # CPU -> GPU
    println("  CPU -> GPU transfer...")
    cpu_data = image.voxel_data
    b_to_gpu = @benchmark begin
        gpu_data = CuArray($cpu_data)
        CUDA.@sync nothing
    end samples=BENCHMARK_SAMPLES

    transfer_rate_to_gpu = data_size / (median(b_to_gpu).time * 1e-9)
    @printf("    Time: %.3f ms (median)\n", median(b_to_gpu).time * 1e-6)
    @printf("    Transfer rate: %.2f MB/s\n", transfer_rate_to_gpu)

    results["cpu_to_gpu_time"] = median(b_to_gpu).time * 1e-9
    results["cpu_to_gpu_rate_mbs"] = transfer_rate_to_gpu

    # GPU -> CPU
    println("  GPU -> CPU transfer...")
    gpu_data = CuArray(cpu_data)
    CUDA.@sync nothing

    b_to_cpu = @benchmark begin
        cpu_result = Array($gpu_data)
        CUDA.@sync nothing
    end samples=BENCHMARK_SAMPLES

    transfer_rate_to_cpu = data_size / (median(b_to_cpu).time * 1e-9)
    @printf("    Time: %.3f ms (median)\n", median(b_to_cpu).time * 1e-6)
    @printf("    Transfer rate: %.2f MB/s\n", transfer_rate_to_cpu)

    results["gpu_to_cpu_time"] = median(b_to_cpu).time * 1e-9
    results["gpu_to_cpu_rate_mbs"] = transfer_rate_to_cpu
    results["data_size_mb"] = data_size

    return results
end
