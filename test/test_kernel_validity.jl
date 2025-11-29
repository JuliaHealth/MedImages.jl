using Test
using MedImages
using MedImages.Utils
using Random

function test_kernel_validity()
    @testset "Kernel Validity vs Interpolations.jl" begin
        # Generate random 3D data
        Random.seed!(42)
        size_src = (32, 32, 32)
        data = rand(Float32, size_src)

        old_spacing = (1.0, 1.0, 1.0)

        # Test cases
        spacings = [
            (0.5, 0.5, 0.5), # Upsample
            (2.0, 2.0, 2.0), # Downsample
            (1.5, 0.8, 1.2)  # Mixed
        ]

        for new_spacing in spacings
            @testset "Spacing $new_spacing" begin
                new_dims = Tuple(Int(ceil(sz * osp / nsp)) for (sz, osp, nsp) in zip(size_src, old_spacing, new_spacing))

                # 1. Fast Kernel (Optimized)
                # resample_kernel_launch takes (data, old_spacing, new_spacing, new_dims, interpolator)
                res_fast = Utils.resample_kernel_launch(data, old_spacing, new_spacing, new_dims, MedImages.Linear_en)

                # 2. Slow Interpolations.jl (Reference)
                # Construct points in physical space
                indices = Utils.get_base_indicies_arr(new_dims)
                # indices is 3xN
                points = similar(indices, Float64)
                for i in 1:size(indices, 2)
                    points[1, i] = (indices[1, i] - 1) * new_spacing[1] + 1.0
                    points[2, i] = (indices[2, i] - 1) * new_spacing[2] + 1.0
                    points[3, i] = (indices[3, i] - 1) * new_spacing[3] + 1.0
                end

                # interpolate_my(points, input, spacing, interp, keep_beginning, extrapolate, use_fast)
                res_slow_flat = Utils.interpolate_my(points, data, old_spacing, MedImages.Linear_en, true, 0.0, false)
                res_slow = reshape(res_slow_flat, new_dims)

                # Convert to Float32 for comparison
                res_slow = Float32.(res_slow)

                # Compare
                # We expect small differences due to implementation details
                diff = abs.(res_fast .- res_slow)
                mean_diff = sum(diff) / length(diff)
                max_diff = maximum(diff)

                println("Spacing $new_spacing: Mean Diff = $mean_diff, Max Diff = $max_diff")

                @test mean_diff < 1e-4
                @test max_diff < 1e-2 # Trilinear vs BSpline(Linear) should be very close

                # 3. Test Generic Interpolate Kernel (interpolate_my use_fast=true)
                # This tests the refactored interpolate_kernel in Utils.jl
                # We reuse the points we calculated manually
                res_generic_fast_flat = Utils.interpolate_my(points, data, old_spacing, MedImages.Linear_en, true, 0.0, true) # use_fast=true
                res_generic_fast = reshape(res_generic_fast_flat, new_dims)
                res_generic_fast = Float32.(res_generic_fast)

                diff_generic = abs.(res_generic_fast .- res_slow)
                mean_diff_generic = sum(diff_generic) / length(diff_generic)
                max_diff_generic = maximum(diff_generic)

                println("Generic Kernel Spacing $new_spacing: Mean Diff = $mean_diff_generic, Max Diff = $max_diff_generic")

                @test mean_diff_generic < 1e-4
                @test max_diff_generic < 1e-2
            end
        end
    end
end

test_kernel_validity()
