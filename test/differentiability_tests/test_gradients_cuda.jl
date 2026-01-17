using Test
using MedImages
using CUDA
using Zygote
using LinearAlgebra
using Dates
using MedImages.MedImage_data_struct

# Skip if no GPU available
if !CUDA.functional()
    @info "CUDA not available, skipping GPU autodiff tests"
    exit(0)
end

@info "CUDA functional - running GPU autodiff tests" device=CUDA.device()

const TEST_MRI = MedImages.MedImage_data_struct.MRI_type
const TEST_SUBTYPE = MedImages.MedImage_data_struct.T1_subtype
const TEST_LINEAR = MedImages.MedImage_data_struct.Linear_en
const TEST_LPI = MedImages.MedImage_data_struct.ORIENTATION_LPI

# Fixed datetime values to avoid Zygote trying to differentiate through Dates.now()
const FIXED_DATE = DateTime(2024, 1, 1)

# Helper function to check if array is on GPU
is_cuda_array(arr) = isa(arr, CuArray)

# Helper to create CPU MedImage
function create_cpu_medimage(data)
    MedImage(
        voxel_data = data,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        image_type = TEST_MRI,
        image_subtype = TEST_SUBTYPE,
        patient_id = "test_cpu",
        date_of_saving = FIXED_DATE,
        acquistion_time = FIXED_DATE
    )
end

# Helper to create GPU MedImage
function create_gpu_medimage(data_cpu)
    data_gpu = CuArray(Float32.(data_cpu))
    MedImage(
        voxel_data = data_gpu,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        image_type = TEST_MRI,
        image_subtype = TEST_SUBTYPE,
        patient_id = "test_gpu",
        date_of_saving = FIXED_DATE,
        acquistion_time = FIXED_DATE
    )
end

@testset "GPU Forward Pass Tests" begin
    # Verify GPU operations work correctly in forward pass

    @testset "resample_to_spacing GPU forward" begin
        data = rand(Float32, 8, 8, 8)
        im = create_gpu_medimage(data)
        resampled = MedImages.resample_to_spacing(im, (2.0, 2.0, 2.0), TEST_LINEAR)
        @test size(resampled.voxel_data) == (4, 4, 4)
        @test is_cuda_array(resampled.voxel_data)
        @info "resample_to_spacing GPU forward pass works"
    end

    @testset "rotate_mi GPU forward" begin
        data = rand(Float32, 8, 8, 8)
        im = create_gpu_medimage(data)
        rotated = MedImages.rotate_mi(im, 1, 45.0, TEST_LINEAR)
        @test size(rotated.voxel_data) == (8, 8, 8)
        @test is_cuda_array(rotated.voxel_data)
        @info "rotate_mi GPU forward pass works"
    end

    @testset "scale_mi GPU forward" begin
        data = rand(Float32, 8, 8, 8)
        im = create_gpu_medimage(data)
        scaled = MedImages.scale_mi(im, 0.5, TEST_LINEAR)
        @test size(scaled.voxel_data) == (4, 4, 4)
        @test is_cuda_array(scaled.voxel_data)
        @info "scale_mi GPU forward pass works"
    end

    @testset "translate_mi GPU forward" begin
        data = rand(Float32, 8, 8, 8)
        im = create_gpu_medimage(data)
        translated = MedImages.translate_mi(im, 3, 1, TEST_LINEAR)
        @test size(translated.voxel_data) == (8, 8, 8)
        @test is_cuda_array(translated.voxel_data)
        @info "translate_mi GPU forward pass works"
    end
end

@testset "GPU Autodiff - Metadata-Only Operations" begin
    # These operations don't use GPU kernels for data transformation
    # They only update metadata, so autodiff works through them

    @testset "translate_mi GPU - gradient exists" begin
        data = rand(Float32, 8, 8, 8)

        function loss(x)
            im = create_gpu_medimage(x)
            translated = MedImages.translate_mi(im, 3, 1, TEST_LINEAR)
            return sum(Array(translated.voxel_data))
        end

        grads = Zygote.gradient(loss, data)
        @test grads[1] !== nothing
        # Translation just changes origin, so gradient should be all ones
        @test all(isapprox.(grads[1], 1.0))
        @info "translate_mi GPU gradient computed successfully"
    end

    @testset "translate_mi GPU vs CPU gradient consistency" begin
        data = rand(Float32, 8, 8, 8)

        function cpu_loss(x)
            im = create_cpu_medimage(x)
            translated = MedImages.translate_mi(im, 3, 1, TEST_LINEAR)
            return sum(translated.voxel_data)
        end

        function gpu_loss(x)
            im = create_gpu_medimage(x)
            translated = MedImages.translate_mi(im, 3, 1, TEST_LINEAR)
            return sum(Array(translated.voxel_data))
        end

        cpu_grads = Zygote.gradient(cpu_loss, data)[1]
        gpu_grads = Zygote.gradient(gpu_loss, data)[1]

        @test isapprox(cpu_grads, gpu_grads, rtol=1e-4)
        @info "translate_mi: GPU and CPU gradients match" max_diff=maximum(abs.(cpu_grads .- gpu_grads))
    end
end

# Note: GPU autodiff for kernel-based operations (resample, rotate, scale) is not yet supported
# due to Enzyme's inability to differentiate through KernelAbstractions GPU kernels.
# See: https://enzyme.mit.edu/index.fcgi/julia/stable/faq/#Activity-of-temporary-storage
#
# Tested Julia versions:
# - Julia 1.10.8: EnzymeMutabilityException (same error)
# - Julia 1.11.6: EnzymeMutabilityException
# The issue is fundamental to how Enzyme handles KernelAbstractions GPU kernels,
# not specific to a Julia version.
#
# Current status:
# - CPU autodiff: Fully working (uses pure Julia loops with Enzyme)
# - GPU autodiff: Limited to metadata-only operations (translate_mi)
#
# To enable GPU autodiff for kernel-based operations would require:
# - Implementing pure CUDA.jl loops instead of KernelAbstractions kernels
# - Or using a different AD framework compatible with KA (e.g., direct Zygote rules without Enzyme)

@testset "GPU Autodiff - Known Limitations" begin
    @testset "KernelAbstractions kernels - documented limitation" begin
        # This test documents that GPU kernel-based autodiff is a known limitation
        # The forward pass works, but backward pass through Enzyme fails

        data = rand(Float32, 4, 4, 4)

        # Forward pass works
        im = create_gpu_medimage(data)
        resampled = MedImages.resample_to_spacing(im, (2.0, 2.0, 2.0), TEST_LINEAR)
        @test is_cuda_array(resampled.voxel_data)

        # Document that gradient through KA kernels is not supported
        # (Enzyme throws EnzymeMutabilityException)
        @info "GPU kernel autodiff limitation documented: Enzyme cannot differentiate through KernelAbstractions kernels"
        @test true  # Documenting the known limitation
    end
end
