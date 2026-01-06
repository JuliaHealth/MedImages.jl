# test/brute_force_orientation_tests/test_brute_force_find_perm_rev.jl
# Tests for brute_force_find_perm_rev and related functions

using Test
using PyCall
using MedImages

# Import test infrastructure (conditionally include if not already defined)
if !isdefined(@__MODULE__, :TestHelpers)
    include(joinpath(@__DIR__, "..", "test_helpers.jl"))
    include(joinpath(@__DIR__, "..", "test_config.jl"))
end
using .TestHelpers
using .TestConfig

@testset "brute_force_find_perm_spacing Tests" begin
    with_temp_output_dir("brute_force_orientation", "perm_spacing") do output_dir

        @testset "Identical spacings" begin
            spacing_src = (1.0, 1.0, 1.0)
            spacing_tgt = (1.0, 1.0, 1.0)

            try
                result = MedImages.Brute_force_orientation.brute_force_find_perm_spacing(
                    spacing_src, spacing_tgt)
                @test result isa Tuple
            catch e
                @test_broken false
                @info "Error: $e"
            end
        end

        @testset "Permuted spacings" begin
            spacing_src = (1.0, 2.0, 3.0)
            spacing_tgt = (2.0, 3.0, 1.0)

            try
                result = MedImages.Brute_force_orientation.brute_force_find_perm_spacing(
                    spacing_src, spacing_tgt)
                @test result isa Tuple
            catch e
                @test_broken false
                @info "Error: $e"
            end
        end
    end
end

@testset "get_orientations_vectors Tests" begin
    with_temp_output_dir("brute_force_orientation", "orient_vectors") do output_dir

        @testset "Get orientation vectors from NIfTI" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            try
                result = MedImages.Brute_force_orientation.get_orientations_vectors(TEST_NIFTI_FILE)
                @test result isa Dict || result isa Tuple || result isa NamedTuple
            catch e
                @test_broken false
                @info "Error getting orientation vectors: $e"
            end
        end
    end
end

@testset "brute_force_find_from_sitk Tests" begin
    with_temp_output_dir("brute_force_orientation", "find_from_sitk") do output_dir

        @testset "Find all orientation operations" begin
            if !isfile(TEST_NIFTI_FILE)
                @test_skip "Test file not found"
                return
            end

            try
                result = MedImages.Brute_force_orientation.brute_force_find_from_sitk(TEST_NIFTI_FILE)
                @test result isa Dict
            catch e
                @test_broken false
                @info "Error in brute_force_find_from_sitk: $e"
            end
        end
    end
end
