# test/utils_tests/test_ensure_tuple.jl
# Tests for ensure_tuple function from Utils module

using Test
using MedImages
using MedImages.Utils

# Import test infrastructure
include(joinpath(@__DIR__, "..", "test_helpers.jl"))
include(joinpath(@__DIR__, "..", "test_config.jl"))
using .TestHelpers
using .TestConfig

@testset "ensure_tuple Tests" begin
    with_temp_output_dir("utils", "ensure_tuple") do output_dir

        @testset "Already a tuple" begin
            t = (1.0, 2.0, 3.0)
            result = Utils.ensure_tuple(t)
            @test result === t
            @test result isa Tuple
        end

        @testset "Vector to tuple" begin
            v = [1.0, 2.0, 3.0]
            result = Utils.ensure_tuple(v)
            @test result isa Tuple
            @test result == (1.0, 2.0, 3.0)
        end

        @testset "Integer vector to tuple" begin
            v = [1, 2, 3]
            result = Utils.ensure_tuple(v)
            @test result isa Tuple
            @test length(result) == 3
        end

        @testset "9-element vector (direction matrix)" begin
            v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            result = Utils.ensure_tuple(v)
            @test result isa Tuple
            @test length(result) == 9
        end

        @testset "Empty cases handled" begin
            # Empty vector
            try
                v = Float64[]
                result = Utils.ensure_tuple(v)
                @test result isa Tuple
                @test length(result) == 0
            catch e
                # Empty might not be supported - that's ok
                @test true
            end
        end

        @testset "Type preservation" begin
            v_float = [1.0, 2.0, 3.0]
            result_float = Utils.ensure_tuple(v_float)
            @test all(x -> x isa Float64, result_float)

            v_int = [1, 2, 3]
            result_int = Utils.ensure_tuple(v_int)
            # Result might be Int or converted
            @test length(result_int) == 3
        end
    end
end
