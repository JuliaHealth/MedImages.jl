"""
Main GPU Benchmark Runner
Orchestrates all benchmarks for MedImages.jl GPU performance testing.

Usage:
    julia --project=. run_gpu_benchmarks.jl [options]

Options:
    --catalog FILE      Path to benchmark catalog (default: benchmark_data/nifti/benchmark_catalog.csv)
    --synthetic         Use synthetic test images instead of real data
    --operations OPS    Comma-separated list of operations to benchmark
                       (interpolate,resample,cross_resample,rotate,crop,pad,orientation,all)
                       Default: all
    --backends BACK     Comma-separated list of backends (cpu,cuda,all)
                       Default: all
    --output DIR        Output directory for results (default: benchmark_results)
    --help              Show this help message
"""

# First load packages from benchmark project
using Printf
using Dates
using ArgParse

# Now activate parent MedImages.jl project to access MedImages
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using MedImages

# Switch back to benchmark project for other dependencies
Pkg.activate(@__DIR__)

# Include benchmark modules
include("benchmark_config.jl")
include("gpu_benchmarks.jl")
include("benchmark_utils.jl")

"""
    ensure_native_array(img::MedImage) -> MedImage

Convert MedImage voxel_data to native Julia Array if it's wrapped in a C++ type.
SimpleITK returns data wrapped in CxxWrap types which KernelAbstractions doesn't recognize.
"""
function ensure_native_array(img::MedImage)
    voxel_data = img.voxel_data

    # Check if the underlying data is not a native Julia Array
    if !(typeof(voxel_data) <: Array)
        # Convert to native Julia Array and update the mutable struct
        img.voxel_data = Array(voxel_data)
    end
    return img
end

"""
Parse command line arguments
"""
function parse_commandline()
    s = ArgParseSettings(description="MedImages.jl GPU Benchmark Runner")

    @add_arg_table! s begin
        "--catalog"
            help = "Path to benchmark catalog CSV"
            arg_type = String
            default = "benchmark_data/nifti/benchmark_catalog.csv"

        "--synthetic"
            help = "Use synthetic test images"
            action = :store_true

        "--operations"
            help = "Operations to benchmark (comma-separated)"
            arg_type = String
            default = "all"

        "--backends"
            help = "Backends to test (comma-separated: cpu,cuda,all)"
            arg_type = String
            default = "all"

        "--output"
            help = "Output directory"
            arg_type = String
            default = "benchmark_results"
    end

    return parse_args(s)
end

"""
    load_or_create_test_images(catalog_file::String, use_synthetic::Bool) -> Dict

Load benchmark images from catalog or create synthetic images.
"""
function load_or_create_test_images(catalog_file::String, use_synthetic::Bool)
    images = Dict{Symbol, Vector{MedImage}}()

    if use_synthetic
        println("Creating synthetic test images...")

        # Create images - use medium and large to avoid OOM with upsampling
        # xlarge (1024x1024x512) * 4x upsample would require ~4TB RAM
        for (size_name, dims) in IMAGE_SIZES
            if size_name in [:medium, :large, :xlarge]  # Include xlarge for comprehensive benchmarks
                println("  Creating $size_name image: $(dims[1])x$(dims[2])x$(dims[3])")
                img = create_synthetic_test_image(dims)
                images[size_name] = [img]
            end
        end

    else
        println("Loading benchmark images from catalog...")

        if !isfile(catalog_file)
            @error "Catalog file not found" file=catalog_file
            @error "Please run: julia convert_dicom_to_nifti.jl"
            exit(1)
        end

        # Load catalog (CSV and DataFrames loaded via benchmark_config.jl)
        catalog = CSV.read(catalog_file, DataFrame)

        # Group by size category
        for size_cat in [:large, :xlarge]
            size_str = string(size_cat)
            filtered = filter(row -> row.size_category == size_str, catalog)

            if nrow(filtered) > 0
                images[size_cat] = MedImage[]

                for row in eachrow(filtered)
                    println("  Loading: $(row.file)")
                    try
                        # Load NIFTI using MedImages
                        # Since NIFTI files don't contain modality info, we assume CT
                        img = MedImages.load_image(replace(row.file, ".nii.gz" => ""), "CT")
                        # Convert to native Julia Array for KernelAbstractions compatibility
                        img = ensure_native_array(img)
                        push!(images[size_cat], img)
                    catch e
                        @warn "Failed to load image" file=row.file exception=e
                    end
                end

                println("  Loaded $(length(images[size_cat])) images for $size_cat")
            else
                @warn "No images found for category" category=size_cat
            end
        end
    end

    return images
end

"""
    run_all_benchmarks(images::Dict, operations::Vector{String}, backends::Vector{String}) -> Vector{BenchmarkResult}

Run all selected benchmarks on provided images.
"""
function run_all_benchmarks(images::Dict, operations::Vector{<:AbstractString}, backends::Vector{<:AbstractString})
    all_results = []

    println("\n" * "="^80)
    println("Starting Benchmark Suite")
    println("="^80)
    println("Operations: $(join(operations, ", "))")
    println("Backends: $(join(backends, ", "))")
    println("="^80)

    # Iterate through each size category
    for (size_cat, img_list) in images
        if isempty(img_list)
            @warn "No images for category" category=size_cat
            continue
        end

        # Use first image from each category
        img = img_list[1]
        dims = size(img.voxel_data)

        println("\n" * "="^80)
        println("Image Size: $size_cat ($(dims[1])x$(dims[2])x$(dims[3]))")
        println("="^80)

        for backend_name in backends
            backend_name_upper = uppercase(backend_name)

            # Skip CUDA if not available
            if backend_name == "cuda" && !CUDA_AVAILABLE
                @warn "CUDA not available, skipping CUDA benchmarks"
                continue
            end

            println("\n" * "-"^80)
            println("Backend: $backend_name_upper")
            println("-"^80)

            # 1. Interpolation kernel benchmarks
            if "interpolate" in operations || "all" in operations
                println("\n### Interpolation Kernel Benchmarks ###")

                for point_count in POINT_COUNTS
                    for interp_method in INTERPOLATION_METHODS
                        result = benchmark_interpolate_kernel(
                            img, point_count, interp_method.enum, backend_name_upper
                        )
                        push!(all_results, result)
                    end
                end
            end

            # 2. Resampling benchmarks
            if "resample" in operations || "all" in operations
                println("\n### Resampling Benchmarks ###")

                for (factor_name, factor) in RESAMPLING_FACTORS
                    # Skip upsampling for xlarge to avoid OOM
                    if size_cat == :xlarge && factor > 1.0
                        println("  Skipping upsampling ($factor_name) for xlarge to avoid OOM")
                        continue
                    end

                    for interp_method in INTERPOLATION_METHODS
                        try
                            result = benchmark_resample_to_spacing(
                                img, factor, interp_method.enum, backend_name_upper
                            )
                            push!(all_results, result)
                        catch e
                            if occursin("OutOfMemory", string(e)) || occursin("CUDA", string(e))
                                @warn "OOM during resampling benchmark, skipping" factor=factor interp=interp_method.name
                            else
                                rethrow(e)
                            end
                        end
                    end
                end
            end

            # 3. Cross-image resampling (only if we have multiple images)
            if ("cross_resample" in operations || "all" in operations) && length(img_list) >= 2
                println("\n### Cross-Image Resampling Benchmarks ###")

                # Use first two images
                source_img = img_list[1]
                target_img = img_list[min(2, length(img_list))]

                for interp_method in INTERPOLATION_METHODS
                    result = benchmark_resample_to_image(
                        source_img, target_img, interp_method.enum, backend_name_upper
                    )
                    push!(all_results, result)
                end
            end

            # 4. Rotation benchmarks
            if "rotate" in operations || "all" in operations
                println("\n### Rotation Benchmarks ###")

                for angle in ROTATION_ANGLES
                    try
                        result = benchmark_rotation(img, angle, backend_name_upper)
                        push!(all_results, result)
                    catch e
                        if occursin("OutOfMemory", string(e)) || occursin("CUDA", string(e))
                            @warn "OOM during rotation benchmark, skipping" angle=angle size=size_cat
                        else
                            rethrow(e)
                        end
                    end
                end
            end

            # 5. Crop benchmarks
            if "crop" in operations || "all" in operations
                println("\n### Crop Benchmarks ###")

                for (ratio_name, ratio) in CROP_PAD_RATIOS
                    result = benchmark_crop(img, ratio, backend_name_upper)
                    push!(all_results, result)
                end
            end

            # 6. Pad benchmarks (CPU only - CUDA has scalar indexing issues in MedImages.jl)
            if ("pad" in operations || "all" in operations) && backend_name != "cuda"
                println("\n### Pad Benchmarks ###")

                for (ratio_name, ratio) in CROP_PAD_RATIOS
                    result = benchmark_pad(img, ratio, backend_name_upper)
                    push!(all_results, result)
                end
            elseif ("pad" in operations || "all" in operations) && backend_name == "cuda"
                println("\n### Pad Benchmarks (skipped - CUDA scalar indexing not supported) ###")
            end

            # 7. Orientation change benchmarks
            if "orientation" in operations || "all" in operations
                println("\n### Orientation Change Benchmarks ###")

                for orientation in ORIENTATION_CODES[1:min(3, length(ORIENTATION_CODES))]
                    try
                        result = benchmark_change_orientation(img, orientation, backend_name_upper)
                        push!(all_results, result)
                    catch e
                        if occursin("OutOfMemory", string(e)) || occursin("CUDA", string(e)) || occursin("Out of GPU memory", string(e))
                            @warn "OOM during orientation benchmark, skipping" orientation=orientation size=size_cat
                            # Clean up GPU memory
                            if backend_name == "cuda" && CUDA_AVAILABLE
                                try
                                    GC.gc()
                                    CUDA.reclaim()
                                catch; end
                            end
                        else
                            rethrow(e)
                        end
                    end
                end
            end
        end
    end

    # Memory transfer benchmarks (CUDA only)
    memory_stats = Dict()
    if "cuda" in backends && CUDA_AVAILABLE && !isempty(images)
        println("\n" * "="^80)
        println("Memory Transfer Benchmarks")
        println("="^80)

        # Use largest available image
        largest_size = :xlarge in keys(images) ? :xlarge : :large
        if largest_size in keys(images) && !isempty(images[largest_size])
            img = images[largest_size][1]
            memory_stats = benchmark_memory_transfer(img)
        end
    end

    return all_results, memory_stats
end

"""
Main execution function
"""
function main()
    println("="^80)
    println("MedImages.jl GPU Benchmark Suite")
    println("="^80)
    println("Start time: $(now())")
    println()

    # Parse arguments
    args = parse_commandline()

    # Print configuration
    print_config()

    # Parse operation list
    operations = split(args["operations"], ",")
    operations = [strip(op) for op in operations]

    # Parse backend list
    backends = split(args["backends"], ",")
    backends = [strip(lowercase(b)) for b in backends]

    if "all" in backends
        backends = ["cpu", "cuda"]
    end

    # Load or create images
    images = load_or_create_test_images(args["catalog"], args["synthetic"])

    if isempty(images)
        @error "No images available for benchmarking"
        exit(1)
    end

    # Run benchmarks
    all_results, memory_stats = run_all_benchmarks(images, operations, backends)

    # Save and report results
    output_dir = args["output"]
    mkpath(output_dir)

    println("\n" * "="^80)
    println("Generating Reports")
    println("="^80)

    # Save CSV
    csv_file = joinpath(output_dir, "results.csv")
    save_results_csv(all_results, csv_file)

    # Generate markdown report
    report_file = joinpath(output_dir, "report.md")
    generate_markdown_report(all_results, report_file, memory_stats=memory_stats)

    # Print summary
    print_summary(all_results)

    # Plot results (optional)
    try
        plot_benchmark_results(all_results)
    catch e
        @warn "Plotting failed" exception=e
    end

    # SimpleITK comparison (if requested)
    if COMPARE_SIMPLEITK
        compare_with_simpleitk(all_results, images)
    end

    println("\n" * "="^80)
    println("Benchmark Complete!")
    println("="^80)
    println("End time: $(now())")
    println("Results saved to: $output_dir")
    println("  - CSV: $csv_file")
    println("  - Report: $report_file")
    println("="^80)
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
