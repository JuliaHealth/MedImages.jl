# test/utils_tests/test_cuda_utils.jl
# Tests for CUDA utility functions from Utils module

using Test
using MedImages

# Import test infrastructure (conditionally include if not already defined)
if !isdefined(@__MODULE__, :TestHelpers)
    include(joinpath(@__DIR__, "..", "test_helpers.jl"))
    include(joinpath(@__DIR__, "..", "test_config.jl"))
end
using .TestHelpers
using .TestConfig

@testset "CUDA Utils Tests" begin
    with_temp_output_dir("utils", "cuda_utils") do output_dir

        @testset "is_cuda_array function" begin
            # Test with regular Julia Array - should return false
            cpu_array = rand(Float32, 10, 10, 10)
            @test MedImages.Utils.is_cuda_array(cpu_array) == false

            # Test with different array types
            int_array = rand(Int32, 5, 5, 5)
            @test MedImages.Utils.is_cuda_array(int_array) == false

            # Test with 1D array
            vector = rand(Float64, 100)
            @test MedImages.Utils.is_cuda_array(vector) == false

            # Test with 2D array
            matrix = rand(Float32, 50, 50)
            @test MedImages.Utils.is_cuda_array(matrix) == false
        end

        @testset "extract_corners function - basic functionality" begin
            # Create a simple 3D array with known corner values
            arr = zeros(Float32, 4, 5, 6)

            # Set corner values
            arr[1, 1, 1] = 1.0f0
            arr[1, 1, 6] = 2.0f0
            arr[1, 5, 1] = 3.0f0
            arr[1, 5, 6] = 4.0f0
            arr[4, 1, 1] = 5.0f0
            arr[4, 1, 6] = 6.0f0
            arr[4, 5, 1] = 7.0f0
            arr[4, 5, 6] = 8.0f0

            corners = MedImages.Utils.extract_corners(arr)

            # Should return 8 corner values
            @test length(corners) == 8

            # Verify corner values are extracted correctly
            @test corners[1] == 1.0f0  # arr[1,1,1]
            @test corners[2] == 2.0f0  # arr[1,1,end]
            @test corners[3] == 3.0f0  # arr[1,end,1]
            @test corners[4] == 4.0f0  # arr[1,end,end]
            @test corners[5] == 5.0f0  # arr[end,1,1]
            @test corners[6] == 6.0f0  # arr[end,1,end]
            @test corners[7] == 7.0f0  # arr[end,end,1]
            @test corners[8] == 8.0f0  # arr[end,end,end]
        end

        @testset "extract_corners function - different dtypes" begin
            # Test with Int32 array
            int_arr = zeros(Int32, 3, 3, 3)
            int_arr[1, 1, 1] = 10
            int_arr[3, 3, 3] = 20
            corners = MedImages.Utils.extract_corners(int_arr)
            @test length(corners) == 8
            @test corners[1] == 10
            @test corners[8] == 20

            # Test with Float64 array
            f64_arr = rand(Float64, 5, 5, 5)
            corners = MedImages.Utils.extract_corners(f64_arr)
            @test length(corners) == 8
            @test corners[1] == f64_arr[1, 1, 1]
            @test corners[8] == f64_arr[end, end, end]
        end

        @testset "extract_corners function - edge cases" begin
            # Minimum size array (1x1x1)
            small_arr = fill(42.0f0, 1, 1, 1)
            corners = MedImages.Utils.extract_corners(small_arr)
            @test length(corners) == 8
            # All corners should be the same value
            @test all(c -> c == 42.0f0, corners)

            # Array with same value everywhere
            uniform_arr = fill(3.14f0, 10, 10, 10)
            corners = MedImages.Utils.extract_corners(uniform_arr)
            @test length(corners) == 8
            @test all(c -> c == 3.14f0, corners)
        end

        @testset "extract_corners function - asymmetric dimensions" begin
            # Very asymmetric array
            arr = rand(Float32, 2, 100, 5)
            corners = MedImages.Utils.extract_corners(arr)
            @test length(corners) == 8
            @test corners[1] == arr[1, 1, 1]
            @test corners[2] == arr[1, 1, end]
            @test corners[3] == arr[1, end, 1]
            @test corners[4] == arr[1, end, end]
            @test corners[5] == arr[end, 1, 1]
            @test corners[6] == arr[end, 1, end]
            @test corners[7] == arr[end, end, 1]
            @test corners[8] == arr[end, end, end]
        end

    end
end
