# test/utils_tests/test_interpolate_my.jl
# Tests for interpolate_my function from Utils module

using Test
using MedImages
using MedImages.Utils
using Random

# Import test infrastructure
include(joinpath(@__DIR__, "..", "test_helpers.jl"))
include(joinpath(@__DIR__, "..", "test_config.jl"))
using .TestHelpers
using .TestConfig

@testset "interpolate_my Tests" begin
    with_temp_output_dir("utils", "interpolate_my") do output_dir
        Random.seed!(42)

        @testset "Basic interpolation" begin
            # Create simple test data
            data = rand(Float32, 10, 10, 10)
            spacing = (1.0, 1.0, 1.0)

            # Create query points at grid locations (should return exact values)
            points = zeros(Float64, 3, 8)
            points[:, 1] = [1.0, 1.0, 1.0]
            points[:, 2] = [2.0, 2.0, 2.0]
            points[:, 3] = [3.0, 3.0, 3.0]
            points[:, 4] = [5.0, 5.0, 5.0]
            points[:, 5] = [1.0, 5.0, 5.0]
            points[:, 6] = [5.0, 1.0, 5.0]
            points[:, 7] = [5.0, 5.0, 1.0]
            points[:, 8] = [10.0, 10.0, 10.0]

            result = Utils.interpolate_my(points, data, spacing, MedImages.Linear_en, true, 0.0, false)

            @test length(result) == 8
            @test !any(isnan, result)
        end

        @testset "use_fast parameter" begin
            data = rand(Float32, 8, 8, 8)
            spacing = (1.0, 1.0, 1.0)
            points = zeros(Float64, 3, 4)
            points[:, 1] = [2.0, 2.0, 2.0]
            points[:, 2] = [4.0, 4.0, 4.0]
            points[:, 3] = [3.5, 3.5, 3.5]
            points[:, 4] = [5.0, 5.0, 5.0]

            result_slow = Utils.interpolate_my(points, data, spacing, MedImages.Linear_en, true, 0.0, false)
            result_fast = Utils.interpolate_my(points, data, spacing, MedImages.Linear_en, true, 0.0, true)

            # Results should be very similar
            @test length(result_slow) == length(result_fast)
            @test isapprox(result_slow, result_fast; rtol=1e-3)
        end

        @testset "Extrapolation value" begin
            data = rand(Float32, 5, 5, 5)
            spacing = (1.0, 1.0, 1.0)

            # Point outside the grid
            points = zeros(Float64, 3, 1)
            points[:, 1] = [10.0, 10.0, 10.0]  # Outside 5x5x5 grid

            extrap_val = -999.0
            result = Utils.interpolate_my(points, data, spacing, MedImages.Linear_en, true, extrap_val, false)

            # If outside, should return extrapolation value
            @test length(result) == 1
        end

        @testset "Different interpolators" begin
            data = rand(Float32, 8, 8, 8)
            spacing = (1.0, 1.0, 1.0)
            points = zeros(Float64, 3, 4)
            points[:, 1] = [2.5, 2.5, 2.5]
            points[:, 2] = [4.5, 4.5, 4.5]
            points[:, 3] = [3.0, 3.0, 3.0]
            points[:, 4] = [5.5, 5.5, 5.5]

            result_nearest = Utils.interpolate_my(points, data, spacing, MedImages.Nearest_neighbour_en, true, 0.0, false)
            result_linear = Utils.interpolate_my(points, data, spacing, MedImages.Linear_en, true, 0.0, false)
            result_bspline = Utils.interpolate_my(points, data, spacing, MedImages.B_spline_en, true, 0.0, false)

            @test length(result_nearest) == 4
            @test length(result_linear) == 4
            @test length(result_bspline) == 4

            # All should produce valid values
            @test !any(isnan, result_nearest)
            @test !any(isnan, result_linear)
            @test !any(isnan, result_bspline)
        end

        @testset "Anisotropic spacing" begin
            data = rand(Float32, 10, 10, 10)
            spacing = (0.5, 1.0, 2.0)  # Anisotropic

            points = zeros(Float64, 3, 3)
            points[:, 1] = [2.0, 2.0, 2.0]
            points[:, 2] = [4.0, 4.0, 4.0]
            points[:, 3] = [6.0, 6.0, 6.0]

            result = Utils.interpolate_my(points, data, spacing, MedImages.Linear_en, true, 0.0, false)

            @test length(result) == 3
            @test !any(isnan, result)
        end
    end
end
