using MedImages
using PyCall
using Dates
using Printf
using Statistics

# Try to use BenchmarkTools if available, otherwise fallback to manual timing
const HAS_BENCHMARKTOOLS = try
    using BenchmarkTools
    true
catch
    false
end

# Check for CUDA
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

# Define paths
const BENCHMARK_DIR = @__DIR__
const TEST_DATA_DIR = joinpath(BENCHMARK_DIR, "..", "test_data")
const NIFTI_FILE = joinpath(TEST_DATA_DIR, "volume-0.nii.gz")

# Setup SimpleITK
const sitk = pyimport("SimpleITK")
const np = pyimport("numpy")

# --- Helper functions for SimpleITK ---

function matrix_from_axis_angle(a)
    ux, uy, uz, theta = a
    c = cos(theta)
    s = sin(theta)
    ci = 1.0 - c
    R = [[ci * ux * ux + c,
            ci * ux * uy - uz * s,
            ci * ux * uz + uy * s],
        [ci * uy * ux + uz * s,
            ci * uy * uy + c,
            ci * uy * uz - ux * s],
        [ci * uz * ux - uy * s,
            ci * uz * uy + ux * s,
            ci * uz * uz + c]]
    return R
end

function get_center_sitk(img)
    width, height, depth = img.GetSize()
    centt = (Int(ceil(width / 2)), Int(ceil(height / 2)), Int(ceil(depth / 2)))
    return img.TransformIndexToPhysicalPoint(centt)
end

function rotation3d_sitk(image, axis, theta)
    theta = np.deg2rad(theta)
    euler_transform = sitk.Euler3DTransform()
    image_center = get_center_sitk(image)
    euler_transform.SetCenter(image_center)

    direction = image.GetDirection()

    if axis == 3
        axis_angle = (direction[3], direction[6], direction[9], theta)
    elseif axis == 2
        axis_angle = (direction[2], direction[5], direction[8], theta)
    elseif axis == 1
        axis_angle = (direction[1], direction[4], direction[7], theta)
    end

    np_rot_mat = matrix_from_axis_angle(axis_angle)
    euler_transform.SetMatrix([np_rot_mat[1][1], np_rot_mat[1][2], np_rot_mat[1][3],
                              np_rot_mat[2][1], np_rot_mat[2][2], np_rot_mat[2][3],
                              np_rot_mat[3][1], np_rot_mat[3][2], np_rot_mat[3][3]])

    interpolator = sitk.sitkLinear
    default_value = 0.0
    return sitk.Resample(image, image, euler_transform, interpolator, default_value)
end

function resample_sitk(image, new_spacing)
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_size = [
        Int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        Int(round(original_size[2] * (original_spacing[2] / new_spacing[2]))),
        Int(round(original_size[3] * (original_spacing[3] / new_spacing[3])))
    ]
    # Explicitly convert to Python list of integers to avoid TypeError with Int64
    # SimpleITK expects uint32 for size
    py_new_size = PyObject((Int(new_size[1]), Int(new_size[2]), Int(new_size[3])))
    return sitk.Resample(image, py_new_size, sitk.Transform(), sitk.sitkLinear,
                         image.GetOrigin(), new_spacing, image.GetDirection(), 0.0, image.GetPixelID())
end

# --- Benchmarking functions ---

function run_manual_benchmark(name, f, args...)
    # Warmup
    f(args...)

    # Run
    times = Float64[]
    for i in 1:10
        t = @elapsed f(args...)
        push!(times, t)
    end
    return mean(times), std(times)
end

function print_result(name, time_medimages, time_sitk)
    ratio = time_medimages / time_sitk
    @printf("%-20s | %10.4f s | %10.4f s | %.2fx\n", name, time_medimages, time_sitk, ratio)
end

function run_benchmarks()
    println("Running benchmarks...")
    println("----------------------------------------------------------------")
    @printf("%-20s | %10s | %10s | %s\n", "Operation", "MedImages", "SimpleITK", "Ratio (MI/SITK)")
    println("----------------------------------------------------------------")

    # 1. Load Image
    t_load_mi, _ = run_manual_benchmark("Load Image", MedImages.load_image, NIFTI_FILE, "CT")
    t_load_sitk, _ = run_manual_benchmark("Load Image", sitk.ReadImage, NIFTI_FILE)
    print_result("Load Image", t_load_mi, t_load_sitk)

    # Prepare for processing
    mi_img = MedImages.load_image(NIFTI_FILE, "CT")
    sitk_img = sitk.ReadImage(NIFTI_FILE)

    # 2. Rotation
    t_rot_mi, _ = run_manual_benchmark("Rotation (90 deg)", MedImages.rotate_mi, mi_img, 3, 90.0, MedImages.Linear_en)
    t_rot_sitk, _ = run_manual_benchmark("Rotation (90 deg)", rotation3d_sitk, sitk_img, 3, 90.0)
    print_result("Rotation", t_rot_mi, t_rot_sitk)

    # 3. Resampling
    new_spacing = (1.5, 1.5, 1.5)
    t_res_mi, _ = run_manual_benchmark("Resampling", MedImages.resample_to_spacing, mi_img, new_spacing, MedImages.Linear_en)
    t_res_sitk, _ = run_manual_benchmark("Resampling", resample_sitk, sitk_img, new_spacing)
    print_result("Resampling", t_res_mi, t_res_sitk)

    if HAS_CUDA
        println("\nRunning GPU benchmarks...")
        # Prepare GPU image
        mi_img_gpu = deepcopy(mi_img)
        mi_img_gpu.voxel_data = CuArray(mi_img.voxel_data)

        # Warmup and run
        t_res_gpu, _ = run_manual_benchmark("Resampling (GPU)", MedImages.resample_to_spacing, mi_img_gpu, new_spacing, MedImages.Linear_en)
        @printf("%-20s | %10.4f s | %10s | %s\n", "Resampling (GPU)", t_res_gpu, "N/A", "N/A")

        # Calculate speedup vs CPU
        speedup = t_res_mi / t_res_gpu
        println("GPU Speedup vs Native CPU: $(round(speedup, digits=2))x")
    else
        println("\nGPU not available, skipping GPU benchmarks.")
    end

    println("----------------------------------------------------------------")
    println("Note: Lower time is better. Ratio < 1.0 means MedImages is faster.")
end

run_benchmarks()
