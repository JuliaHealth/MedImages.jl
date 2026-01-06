# test/kernel_validity_tests/test_kernel_validity.jl
# Tests for kernel validity comparing fast kernels with Interpolations.jl reference

using Test
using MedImages
using MedImages.Utils
using Random

# Import test infrastructure (conditionally include if not already defined)
if !isdefined(@__MODULE__, :TestHelpers)
    include(joinpath(@__DIR__, "..", "test_helpers.jl"))
    include(joinpath(@__DIR__, "..", "test_config.jl"))
end
using .TestHelpers
using .TestConfig

@testset "Kernel Validity Tests" begin
    with_temp_output_dir("kernel_validity", "kernel_validity") do output_dir
        # Generate random 3D data with fixed seed for reproducibility
        Random.seed!(42)
        size_src = (32, 32, 32)
        data = rand(Float32, size_src)
        old_spacing = (1.0, 1.0, 1.0)

        # Test cases for different resampling scenarios
        spacing_cases = [
            ((0.5, 0.5, 0.5), "Upsample"),
            ((2.0, 2.0, 2.0), "Downsample"),
            ((1.5, 0.8, 1.2), "Mixed")
        ]

        @testset "Fast Kernel vs Interpolations.jl Reference" begin
            for (new_spacing, case_name) in spacing_cases
                @testset "$case_name: spacing=$new_spacing" begin
                    new_dims = Tuple(Int(ceil(sz * osp / nsp))
                                     for (sz, osp, nsp) in zip(size_src, old_spacing, new_spacing))

                    # 1. Fast Kernel (Optimized)
                    res_fast = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                            new_dims, MedImages.Linear_en)

                    # 2. Slow Interpolations.jl (Reference)
                    indices = Utils.get_base_indicies_arr(new_dims)
                    points = similar(indices, Float64)
                    for i in 1:size(indices, 2)
                        points[1, i] = (indices[1, i] - 1) * new_spacing[1] + 1.0
                        points[2, i] = (indices[2, i] - 1) * new_spacing[2] + 1.0
                        points[3, i] = (indices[3, i] - 1) * new_spacing[3] + 1.0
                    end

                    res_slow_flat = Utils.interpolate_my(points, data, old_spacing,
                                                          MedImages.Linear_en, true, 0.0, false)
                    res_slow = reshape(res_slow_flat, new_dims)
                    res_slow = Float32.(res_slow)

                    # Compare
                    diff = abs.(res_fast .- res_slow)
                    mean_diff = sum(diff) / length(diff)
                    max_diff = maximum(diff)

                    @info "$case_name: Mean Diff = $mean_diff, Max Diff = $max_diff"

                    @test mean_diff < 1e-4
                    @test max_diff < 1e-2
                end
            end
        end

        @testset "Generic Interpolate Kernel (use_fast=true)" begin
            for (new_spacing, case_name) in spacing_cases
                @testset "$case_name: spacing=$new_spacing" begin
                    new_dims = Tuple(Int(ceil(sz * osp / nsp))
                                     for (sz, osp, nsp) in zip(size_src, old_spacing, new_spacing))

                    # Calculate points
                    indices = Utils.get_base_indicies_arr(new_dims)
                    points = similar(indices, Float64)
                    for i in 1:size(indices, 2)
                        points[1, i] = (indices[1, i] - 1) * new_spacing[1] + 1.0
                        points[2, i] = (indices[2, i] - 1) * new_spacing[2] + 1.0
                        points[3, i] = (indices[3, i] - 1) * new_spacing[3] + 1.0
                    end

                    # Reference (slow)
                    res_slow_flat = Utils.interpolate_my(points, data, old_spacing,
                                                          MedImages.Linear_en, true, 0.0, false)
                    res_slow = reshape(res_slow_flat, new_dims)
                    res_slow = Float32.(res_slow)

                    # Generic fast kernel
                    res_generic_fast_flat = Utils.interpolate_my(points, data, old_spacing,
                                                                  MedImages.Linear_en, true, 0.0, true)
                    res_generic_fast = reshape(res_generic_fast_flat, new_dims)
                    res_generic_fast = Float32.(res_generic_fast)

                    # Compare
                    diff_generic = abs.(res_generic_fast .- res_slow)
                    mean_diff_generic = sum(diff_generic) / length(diff_generic)
                    max_diff_generic = maximum(diff_generic)

                    @info "Generic Kernel $case_name: Mean Diff = $mean_diff_generic, Max Diff = $max_diff_generic"

                    @test mean_diff_generic < 1e-4
                    @test max_diff_generic < 1e-2
                end
            end
        end
    end
end

@testset "Interpolator Consistency Tests" begin
    with_temp_output_dir("kernel_validity", "interpolator_consistency") do output_dir
        Random.seed!(123)
        size_src = (16, 16, 16)
        data = rand(Float32, size_src)
        old_spacing = (1.0, 1.0, 1.0)
        new_spacing = (0.75, 0.75, 0.75)
        new_dims = Tuple(Int(ceil(sz * osp / nsp))
                         for (sz, osp, nsp) in zip(size_src, old_spacing, new_spacing))

        @testset "Nearest neighbor produces valid output" begin
            result = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                   new_dims, MedImages.Nearest_neighbour_en)
            @test size(result) == new_dims
            @test !any(isnan, result)
            @test !any(isinf, result)
        end

        @testset "Linear produces valid output" begin
            result = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                   new_dims, MedImages.Linear_en)
            @test size(result) == new_dims
            @test !any(isnan, result)
            @test !any(isinf, result)
        end

        @testset "B-spline produces valid output" begin
            result = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                   new_dims, MedImages.B_spline_en)
            @test size(result) == new_dims
            @test !any(isnan, result)
            @test !any(isinf, result)
        end
    end
end

@testset "Edge Case Tests" begin
    with_temp_output_dir("kernel_validity", "edge_cases") do output_dir
        @testset "Single voxel resampling" begin
            data = rand(Float32, 2, 2, 2)
            old_spacing = (1.0, 1.0, 1.0)
            new_spacing = (0.5, 0.5, 0.5)
            new_dims = (4, 4, 4)

            result = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                   new_dims, MedImages.Linear_en)
            @test size(result) == new_dims
        end

        @testset "Large downsampling" begin
            data = rand(Float32, 64, 64, 64)
            old_spacing = (1.0, 1.0, 1.0)
            new_spacing = (4.0, 4.0, 4.0)
            new_dims = (16, 16, 16)

            result = Utils.resample_kernel_launch(data, old_spacing, new_spacing,
                                                   new_dims, MedImages.Linear_en)
            @test size(result) == new_dims
        end
    end
end
