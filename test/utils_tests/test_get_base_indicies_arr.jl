# test/utils_tests/test_get_base_indicies_arr.jl
# Tests for get_base_indicies_arr function from Utils module

using Test
using MedImages
using MedImages.Utils

# Import test infrastructure
include(joinpath(@__DIR__, "..", "test_helpers.jl"))
include(joinpath(@__DIR__, "..", "test_config.jl"))
using .TestHelpers
using .TestConfig

@testset "get_base_indicies_arr Tests" begin
    with_temp_output_dir("utils", "get_base_indicies_arr") do output_dir

        @testset "Basic functionality" begin
            dims = (3, 4, 5)
            result = Utils.get_base_indicies_arr(dims)

            # Result should be 3xN where N = prod(dims)
            @test size(result, 1) == 3
            @test size(result, 2) == prod(dims)
        end

        @testset "Small dimensions" begin
            dims = (2, 2, 2)
            result = Utils.get_base_indicies_arr(dims)

            @test size(result, 2) == 8  # 2*2*2 = 8 points
        end

        @testset "Single dimension" begin
            dims = (1, 1, 1)
            result = Utils.get_base_indicies_arr(dims)

            @test size(result, 2) == 1
        end

        @testset "Asymmetric dimensions" begin
            dims = (10, 5, 3)
            result = Utils.get_base_indicies_arr(dims)

            @test size(result, 2) == 150  # 10*5*3 = 150
        end

        @testset "Index values are valid" begin
            dims = (3, 4, 5)
            result = Utils.get_base_indicies_arr(dims)

            # All indices should be within valid ranges
            for i in 1:size(result, 2)
                @test 1 <= result[1, i] <= dims[1]
                @test 1 <= result[2, i] <= dims[2]
                @test 1 <= result[3, i] <= dims[3]
            end
        end

        @testset "All grid points covered" begin
            dims = (2, 3, 2)
            result = Utils.get_base_indicies_arr(dims)

            # Convert to set of tuples to check uniqueness and coverage
            points = Set{Tuple{Int,Int,Int}}()
            for i in 1:size(result, 2)
                push!(points, (Int(result[1, i]), Int(result[2, i]), Int(result[3, i])))
            end

            # Should have exactly prod(dims) unique points
            @test length(points) == prod(dims)
        end
    end
end
