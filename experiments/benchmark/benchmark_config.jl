"""
Benchmark Configuration
Defines all parameters and settings for GPU benchmarking of MedImages.jl.
"""

using MedImages
using CSV
using DataFrames
using Random
using Dates

# Image size categories (priority: large and xlarge per user requirements)
const IMAGE_SIZES = Dict(
    :small => (128, 128, 64),
    :medium => (256, 256, 128),
    :large => (512, 512, 256),      # Priority
    :xlarge => (1024, 1024, 512)    # Priority
)

# Interpolation methods to benchmark
const INTERPOLATION_METHODS = [
    (name="Nearest Neighbor", enum=MedImages.Nearest_neighbour_en),
    (name="Linear", enum=MedImages.Linear_en)
    # Note: B-spline only works with CPU backend (slow path)
]

# Spacing variations for resampling tests
const SPACING_CONFIGS = Dict(
    :isotropic => (1.0, 1.0, 1.0),
    :anisotropic_ct => (0.7, 0.7, 5.0),    # Typical clinical CT
    :anisotropic_mri => (1.0, 1.0, 3.0)    # Typical MRI
)

# Resampling factors - avoid large upsampling to prevent OOM
# A 512x512x256 image upsampled 2x = 1024x1024x512 = 2GB (manageable)
# A 512x512x256 image upsampled 4x = 2048x2048x1024 = 16GB (too large)
const RESAMPLING_FACTORS = Dict(
    :upsample_1_5x => 1.5,
    :downsample_2x => 0.5,
    :downsample_4x => 0.25
)

# Point counts for direct interpolation kernel benchmarks
const POINT_COUNTS = [
    100_000,      # 100K points
    1_000_000     # 1M points (skip 10M to reduce benchmark time)
]

# Orientation codes for cross-image resampling tests
const ORIENTATION_CODES = [
    MedImages.ORIENTATION_LAS,
    MedImages.ORIENTATION_RAS,
    MedImages.ORIENTATION_RPI,
    MedImages.ORIENTATION_LPS
]

# Rotation angles (degrees)
const ROTATION_ANGLES = [90, 180, 270]

# Crop/Pad sizes (relative to image size)
const CROP_PAD_RATIOS = Dict(
    :small => 0.5,   # Crop/pad by 50%
    :medium => 0.3,  # Crop/pad by 30%
    :large => 0.1    # Crop/pad by 10%
)

# Backend configurations
struct BackendConfig
    name::String
    use_gpu::Bool
end

const BACKENDS = [
    BackendConfig("CPU", false),
    BackendConfig("CUDA", true)
]

# BenchmarkTools configuration
const BENCHMARK_SAMPLES = 10        # Number of samples for each benchmark
const BENCHMARK_SECONDS = 2.0       # Minimum time to run each benchmark
const WARMUP_ITERATIONS = 3         # GPU warmup iterations

# Memory profiling configuration
const PROFILE_MEMORY = true
const MEMORY_SAMPLE_INTERVAL = 0.1  # seconds

# Output configuration
const RESULTS_DIR = "benchmark_results"
const REPORT_FILE = joinpath(RESULTS_DIR, "report.md")
const CSV_FILE = joinpath(RESULTS_DIR, "results.csv")
const MEMORY_FILE = joinpath(RESULTS_DIR, "memory_profile.txt")

# SimpleITK comparison configuration
const COMPARE_SIMPLEITK = true      # User requested SimpleITK comparison

"""
    get_benchmark_images(catalog_file::String; size_categories::Vector{Symbol}=[:large, :xlarge]) -> Dict

Load benchmark images from catalog, filtered by size category.

# Arguments
- `catalog_file`: Path to benchmark catalog CSV
- `size_categories`: Which size categories to include (default: [:large, :xlarge])

# Returns
Dictionary mapping size category to list of image paths
"""
function get_benchmark_images(catalog_file::String; size_categories::Vector{Symbol}=[:large, :xlarge])
    if !isfile(catalog_file)
        @error "Catalog file not found" file=catalog_file
        return Dict{Symbol, Vector{String}}()
    end

    catalog = CSV.read(catalog_file, DataFrame)

    images = Dict{Symbol, Vector{String}}()

    for cat in size_categories
        cat_str = string(cat)
        filtered = filter(row -> row.size_category == cat_str, catalog)

        if nrow(filtered) > 0
            images[cat] = filtered.file
            println("Loaded $(nrow(filtered)) images for category: $cat")
        else
            @warn "No images found for category" category=cat
        end
    end

    return images
end

"""
    create_synthetic_test_image(dims::Tuple{Int,Int,Int}, spacing::Tuple{Float64,Float64,Float64}=(1.0, 1.0, 1.0)) -> MedImage

Create a synthetic test image for benchmarking when real data is not available.
"""
function create_synthetic_test_image(dims::Tuple{Int,Int,Int},
                                    spacing::Tuple{Float64,Float64,Float64}=(1.0, 1.0, 1.0))
    # Create random data with some structure (sphere in center)
    data = zeros(Float32, dims)
    center = [d ÷ 2 for d in dims]
    radius = minimum(dims) ÷ 4

    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        dist = sqrt((i-center[1])^2 + (j-center[2])^2 + (k-center[3])^2)
        if dist < radius
            data[i,j,k] = Float32(1000.0 * (1.0 - dist/radius) + randn() * 10.0)
        else
            data[i,j,k] = Float32(randn() * 10.0)
        end
    end

    # Access enum values via the internal module
    MedImage_data_struct = MedImages.MedImage_data_struct

    # Create MedImage using keyword constructor
    med_image = MedImages.MedImage(
        voxel_data=data,
        origin=(0.0, 0.0, 0.0),
        spacing=spacing,
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        image_type=MedImage_data_struct.CT_type,
        image_subtype=MedImage_data_struct.CT_subtype,
        date_of_saving=DateTime(2026, 1, 1),
        acquistion_time=DateTime(2026, 1, 1),
        patient_id="SYNTHETIC",
        current_device=MedImage_data_struct.CPU_current_device,
        study_uid="",
        patient_uid="",
        series_uid="",
        study_description="Synthetic test image",
        legacy_file_name="",
        display_data=Dict{Any,Any}(),
        clinical_data=Dict{Any,Any}(),
        is_contrast_administered=false,
        metadata=Dict{Any,Any}()
    )

    return med_image
end

"""
    print_config()

Print current benchmark configuration.
"""
function print_config()
    println("="^80)
    println("Benchmark Configuration")
    println("="^80)

    println("\nImage Sizes:")
    for (name, dims) in IMAGE_SIZES
        println("  $name: $(dims[1]) x $(dims[2]) x $(dims[3])")
    end

    println("\nInterpolation Methods:")
    for method in INTERPOLATION_METHODS
        println("  - $(method.name)")
    end

    println("\nSpacing Configurations:")
    for (name, spacing) in SPACING_CONFIGS
        println("  $name: $(spacing[1]) x $(spacing[2]) x $(spacing[3]) mm")
    end

    println("\nResampling Factors:")
    for (name, factor) in RESAMPLING_FACTORS
        println("  $name: $(factor)x")
    end

    println("\nPoint Counts: $(join(POINT_COUNTS, ", "))")

    println("\nBackends:")
    for backend in BACKENDS
        println("  - $(backend.name) (GPU: $(backend.use_gpu))")
    end

    println("\nBenchmark Settings:")
    println("  Samples: $BENCHMARK_SAMPLES")
    println("  Min time: $(BENCHMARK_SECONDS)s")
    println("  Warmup iterations: $WARMUP_ITERATIONS")

    println("\nOutput:")
    println("  Results directory: $RESULTS_DIR")
    println("  Compare SimpleITK: $COMPARE_SIMPLEITK")

    println("="^80)
end
