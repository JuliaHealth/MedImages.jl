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

@testset "GPU Autodiff - Kernel Operations (Direct GPU AD)" begin
    @testset "rotate_mi GPU gradient" begin
        data = rand(Float32, 8, 8, 8)
        
        function rotate_loss(x)
            im = create_gpu_medimage(x)
            rotated = MedImages.rotate_mi(im, 1, 10.0, TEST_LINEAR)
            return sum(Array(rotated.voxel_data))
        end
        
        println("Computing rotate_mi GPU gradient...")
        grads = Zygote.gradient(rotate_loss, data)
        @test grads[1] !== nothing
        @test !all(isapprox.(grads[1], 0.0, atol=1e-5))
        @info "rotate_mi GPU gradient computed successfully"
    end

    @testset "resample_to_spacing GPU gradient" begin
        data = rand(Float32, 8, 8, 8)
        
        function resample_loss(x)
            im = create_gpu_medimage(x)
            resampled = MedImages.resample_to_spacing(im, (2.0, 2.0, 2.0), TEST_LINEAR)
            return sum(Array(resampled.voxel_data))
        end
        
        println("Computing resample_to_spacing GPU gradient...")
        grads = Zygote.gradient(resample_loss, data)
        @test grads[1] !== nothing
        @test !all(isapprox.(grads[1], 0.0, atol=1e-5))
        @info "resample_to_spacing GPU gradient computed successfully"
    end
end
