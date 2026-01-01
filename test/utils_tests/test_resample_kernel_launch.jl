# test/utils_tests/test_resample_kernel_launch.jl
# Tests for resample_kernel_launch function from Utils module

using Test
using MedImages
using MedImages.Utils
using Random

# Import test infrastructure
include(joinpath(@__DIR__, "..", "test_helpers.jl"))
include(joinpath(@__DIR__, "..", "test_config.jl"))
using .TestHelpers
using .TestConfig

@testset "resample_kernel_launch Tests" begin
    with_temp_output_dir("utils", "resample_kernel_launch") do output_dir
        Random.seed!(42)

        @testset "Basic resampling" begin
            data = rand(Float32, 16, 16, 16)
            old_spacing = (1.0, 1.0, 1.0)
            new_spacing = (0.5, 0.5, 0.5)
            new_dims = (32, 32, 32)

            result = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                   new_dims, MedImages.Linear_en)

            @test size(result) == new_dims
            @test !any(isnan, result)
            @test !any(isinf, result)
        end

        @testset "Downsampling" begin
            data = rand(Float32, 32, 32, 32)
            old_spacing = (1.0, 1.0, 1.0)
            new_spacing = (2.0, 2.0, 2.0)
            new_dims = (16, 16, 16)

            result = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                   new_dims, MedImages.Linear_en)

            @test size(result) == new_dims
        end

        @testset "Anisotropic resampling" begin
            data = rand(Float32, 20, 20, 20)
            old_spacing = (1.0, 1.0, 1.0)
            new_spacing = (0.5, 1.0, 2.0)
            new_dims = (40, 20, 10)

            result = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                   new_dims, MedImages.Linear_en)

            @test size(result) == new_dims
        end

        @testset "All interpolators work" begin
            data = rand(Float32, 10, 10, 10)
            old_spacing = (1.0, 1.0, 1.0)
            new_spacing = (0.75, 0.75, 0.75)
            new_dims = (14, 14, 14)

            for interp in INTERPOLATORS
                result = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                       new_dims, interp)
                @test size(result) == new_dims
                @test !any(isnan, result)
            end
        end

        @testset "Identity resampling" begin
            data = rand(Float32, 10, 10, 10)
            old_spacing = (1.0, 1.0, 1.0)
            new_spacing = (1.0, 1.0, 1.0)
            new_dims = (10, 10, 10)

            result = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                   new_dims, MedImages.Linear_en)

            @test size(result) == new_dims
            # Values should be approximately the same for identity resampling
            @test isapprox(result, data; rtol=1e-4)
        end

        @testset "Output type matches input type" begin
            data = rand(Float32, 8, 8, 8)
            old_spacing = (1.0, 1.0, 1.0)
            new_spacing = (0.5, 0.5, 0.5)
            new_dims = (16, 16, 16)

            result = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                   new_dims, MedImages.Linear_en)

            @test eltype(result) == Float32
        end
    end
end
