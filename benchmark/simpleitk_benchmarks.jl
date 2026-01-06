"""
SimpleITK Benchmark Comparison
Compares MedImages.jl performance against SimpleITK (Python) for equivalent operations.
"""

using PyCall
using Statistics
using Printf
using Dates

# Import SimpleITK
const sitk = pyimport("SimpleITK")
const np = pyimport("numpy")
const time_module = pyimport("time")

# Helper to convert Julia arrays to proper Python types
function to_py_size(v)
    # SimpleITK expects a Python list of unsigned integers
    # Use py"[]" to create Python list directly
    py"[int(x) for x in $v]"
end

"""
    SimpleITKBenchmarkResult

Result structure for SimpleITK benchmarks.
"""
struct SimpleITKBenchmarkResult
    operation::String
    image_size::String
    time_mean::Float64
    time_median::Float64
    time_std::Float64
    timestamp::DateTime
end

"""
    create_simpleitk_image(dims::Tuple, spacing::Tuple, data::Array)

Create a SimpleITK image from Julia array.
"""
function create_simpleitk_image(dims::Tuple, spacing::Tuple, data::Array)
    # Convert to numpy array (SimpleITK expects [z, y, x] order)
    np_data = np.array(permutedims(Float32.(data), (3, 2, 1)))

    # Create SimpleITK image
    img = sitk.GetImageFromArray(np_data)
    img.SetSpacing(collect(Float64.(spacing)))
    img.SetOrigin((0.0, 0.0, 0.0))

    return img
end

"""
    benchmark_simpleitk_resample(img, new_spacing, interpolator, num_runs=10)

Benchmark SimpleITK resampling operation.
"""
function benchmark_simpleitk_resample(img, new_spacing::Tuple, interpolator::String, num_runs::Int=10)
    # Setup resampler
    resampler = sitk.ResampleImageFilter()

    # Get original properties
    orig_size = img.GetSize()
    orig_spacing = img.GetSpacing()

    # Calculate new size
    new_size = Tuple(Int(ceil(orig_size[i] * orig_spacing[i] / new_spacing[i])) for i in 1:3)

    resampler.SetOutputSpacing(collect(Float64.(new_spacing)))
    resampler.SetSize(to_py_size(collect(new_size)))
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())

    if interpolator == "nearest"
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else
        resampler.SetInterpolator(sitk.sitkLinear)
    end

    # Warmup
    resampler.Execute(img)

    # Benchmark
    times = Float64[]
    for _ in 1:num_runs
        start = time_module.perf_counter()
        result = resampler.Execute(img)
        elapsed = time_module.perf_counter() - start
        push!(times, elapsed)
    end

    return times
end

"""
    benchmark_simpleitk_rotation(img, angle_degrees, axis, num_runs=10)

Benchmark SimpleITK rotation operation.
"""
function benchmark_simpleitk_rotation(img, angle_degrees::Float64, axis::Int, num_runs::Int=10)
    # Get image center
    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()

    center = Tuple(origin[i] + (size[i] * spacing[i]) / 2.0 for i in 1:3)

    # Create rotation transform
    transform = sitk.Euler3DTransform()
    transform.SetCenter(collect(Float64.(center)))

    angle_rad = deg2rad(angle_degrees)
    if axis == 1
        transform.SetRotation(angle_rad, 0.0, 0.0)
    elseif axis == 2
        transform.SetRotation(0.0, angle_rad, 0.0)
    else
        transform.SetRotation(0.0, 0.0, angle_rad)
    end

    # Setup resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(transform)

    # Warmup
    resampler.Execute(img)

    # Benchmark
    times = Float64[]
    for _ in 1:num_runs
        start = time_module.perf_counter()
        result = resampler.Execute(img)
        elapsed = time_module.perf_counter() - start
        push!(times, elapsed)
    end

    return times
end

"""
    benchmark_simpleitk_crop(img, crop_start, crop_size, num_runs=10)

Benchmark SimpleITK crop (extract) operation.
"""
function benchmark_simpleitk_crop(img, crop_start::Tuple, crop_size::Tuple, num_runs::Int=10)
    # Warmup
    result = sitk.Extract(img, to_py_size(collect(crop_size)), to_py_size(collect(crop_start)))

    # Benchmark
    times = Float64[]
    for _ in 1:num_runs
        start = time_module.perf_counter()
        result = sitk.Extract(img, to_py_size(collect(crop_size)), to_py_size(collect(crop_start)))
        elapsed = time_module.perf_counter() - start
        push!(times, elapsed)
    end

    return times
end

"""
    benchmark_simpleitk_pad(img, pad_lower, pad_upper, pad_value, num_runs=10)

Benchmark SimpleITK padding operation.
"""
function benchmark_simpleitk_pad(img, pad_lower::Tuple, pad_upper::Tuple, pad_value::Float64, num_runs::Int=10)
    # Warmup
    result = sitk.ConstantPad(img, to_py_size(collect(pad_lower)), to_py_size(collect(pad_upper)), pad_value)

    # Benchmark
    times = Float64[]
    for _ in 1:num_runs
        start = time_module.perf_counter()
        result = sitk.ConstantPad(img, to_py_size(collect(pad_lower)), to_py_size(collect(pad_upper)), pad_value)
        elapsed = time_module.perf_counter() - start
        push!(times, elapsed)
    end

    return times
end

"""
    run_simpleitk_benchmarks(data::Array, spacing::Tuple, image_size_name::String)

Run all SimpleITK benchmarks on the given data.
"""
function run_simpleitk_benchmarks(data::Array, spacing::Tuple, image_size_name::String)
    results = SimpleITKBenchmarkResult[]
    dims = size(data)

    println("\n### SimpleITK Benchmarks for $image_size_name ###")

    # Create SimpleITK image
    img = create_simpleitk_image(dims, spacing, data)

    # 1. Resample benchmarks
    println("\n  Resampling benchmarks:")

    # Downsample 2x
    new_spacing_down = (spacing[1] * 2.0, spacing[2] * 2.0, spacing[3] * 2.0)

    for (interp_name, interp_type) in [("Nearest", "nearest"), ("Linear", "linear")]
        times = benchmark_simpleitk_resample(img, new_spacing_down, interp_type)
        @printf("    Resample (downsample 2x, %s): %.3f ms (median)\n", interp_name, median(times) * 1000)
        push!(results, SimpleITKBenchmarkResult(
            "resample_downsample_2x_$interp_type",
            image_size_name,
            mean(times),
            median(times),
            std(times),
            now()
        ))
    end

    # Upsample 2x
    new_spacing_up = (spacing[1] / 2.0, spacing[2] / 2.0, spacing[3] / 2.0)

    for (interp_name, interp_type) in [("Nearest", "nearest"), ("Linear", "linear")]
        times = benchmark_simpleitk_resample(img, new_spacing_up, interp_type)
        @printf("    Resample (upsample 2x, %s): %.3f ms (median)\n", interp_name, median(times) * 1000)
        push!(results, SimpleITKBenchmarkResult(
            "resample_upsample_2x_$interp_type",
            image_size_name,
            mean(times),
            median(times),
            std(times),
            now()
        ))
    end

    # 2. Rotation benchmarks
    println("\n  Rotation benchmarks:")
    for angle in [90.0, 180.0, 270.0]
        times = benchmark_simpleitk_rotation(img, angle, 3)  # Rotate around Z axis
        @printf("    Rotate %.0f degrees: %.3f ms (median)\n", angle, median(times) * 1000)
        push!(results, SimpleITKBenchmarkResult(
            "rotate_$(Int(angle))deg",
            image_size_name,
            mean(times),
            median(times),
            std(times),
            now()
        ))
    end

    # 3. Crop benchmarks
    println("\n  Crop benchmarks:")
    crop_offset = (dims[1] ÷ 4, dims[2] ÷ 4, dims[3] ÷ 4)
    crop_size = (dims[1] ÷ 2, dims[2] ÷ 2, dims[3] ÷ 2)

    times = benchmark_simpleitk_crop(img, crop_offset, crop_size)
    @printf("    Crop 50%%: %.3f ms (median)\n", median(times) * 1000)
    push!(results, SimpleITKBenchmarkResult(
        "crop_50pct",
        image_size_name,
        mean(times),
        median(times),
        std(times),
        now()
    ))

    # 4. Pad benchmarks
    println("\n  Pad benchmarks:")
    pad_amount = (dims[1] ÷ 4, dims[2] ÷ 4, dims[3] ÷ 4)

    times = benchmark_simpleitk_pad(img, pad_amount, pad_amount, 0.0)
    @printf("    Pad 50%%: %.3f ms (median)\n", median(times) * 1000)
    push!(results, SimpleITKBenchmarkResult(
        "pad_50pct",
        image_size_name,
        mean(times),
        median(times),
        std(times),
        now()
    ))

    return results
end

"""
    compare_with_simpleitk_full(medimages_results::Vector, synthetic_data::Dict)

Run comprehensive SimpleITK comparison and generate comparison table.
"""
function compare_with_simpleitk_full(medimages_results::Vector, synthetic_data::Dict)
    println("\n" * "="^80)
    println("SimpleITK Comparison Benchmarks")
    println("="^80)

    all_sitk_results = SimpleITKBenchmarkResult[]

    for (size_name, img) in synthetic_data
        data = img.voxel_data
        spacing = img.spacing

        sitk_results = run_simpleitk_benchmarks(data, spacing, string(size_name))
        append!(all_sitk_results, sitk_results)
    end

    return all_sitk_results
end

"""
    generate_comparison_table(medimages_results::Vector, sitk_results::Vector)

Generate a comparison table between MedImages.jl and SimpleITK.
"""
function generate_comparison_table(medimages_results::Vector, sitk_results::Vector)
    println("\n" * "="^80)
    println("MedImages.jl vs SimpleITK Comparison")
    println("="^80)

    # Header
    println()
    @printf("%-40s | %-15s | %-15s | %-15s | %-10s\n",
            "Operation", "MedImages CPU", "MedImages CUDA", "SimpleITK", "Speedup")
    println("-"^100)

    # Group MedImages results by operation
    operations = Dict{String, Dict{String, Float64}}()

    for r in medimages_results
        op_key = "$(r.operation)_$(r.image_size)"
        if !haskey(operations, op_key)
            operations[op_key] = Dict{String, Float64}()
        end
        operations[op_key][r.backend] = r.time_median * 1000  # Convert to ms
    end

    # Match with SimpleITK results
    comparison_data = []

    for sitk_r in sitk_results
        op_name = sitk_r.operation
        size_name = sitk_r.image_size
        sitk_time = sitk_r.time_median * 1000  # ms

        # Try to find matching MedImages result
        med_cpu_time = nothing
        med_cuda_time = nothing

        for r in medimages_results
            if contains(r.name, split(op_name, "_")[1]) && r.image_size == size_name
                if r.backend == "CPU"
                    med_cpu_time = r.time_median * 1000
                elseif r.backend == "CUDA"
                    med_cuda_time = r.time_median * 1000
                end
            end
        end

        # Calculate speedups
        cpu_speedup = med_cpu_time !== nothing ? sitk_time / med_cpu_time : NaN
        cuda_speedup = med_cuda_time !== nothing ? sitk_time / med_cuda_time : NaN

        push!(comparison_data, (
            operation = "$op_name ($size_name)",
            cpu_time = med_cpu_time,
            cuda_time = med_cuda_time,
            sitk_time = sitk_time,
            cuda_speedup = cuda_speedup
        ))
    end

    # Print comparison table
    for row in comparison_data
        cpu_str = row.cpu_time !== nothing ? @sprintf("%.3f ms", row.cpu_time) : "N/A"
        cuda_str = row.cuda_time !== nothing ? @sprintf("%.3f ms", row.cuda_time) : "N/A"
        sitk_str = @sprintf("%.3f ms", row.sitk_time)
        speedup_str = !isnan(row.cuda_speedup) ? @sprintf("%.1fx", row.cuda_speedup) : "N/A"

        @printf("%-40s | %-15s | %-15s | %-15s | %-10s\n",
                row.operation[1:min(40, length(row.operation))],
                cpu_str, cuda_str, sitk_str, speedup_str)
    end

    println("-"^100)

    return comparison_data
end
