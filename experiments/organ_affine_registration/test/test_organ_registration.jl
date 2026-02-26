using Test
using Lux
using Random
using MedImages
using MedImages.MedImage_data_struct
using KernelAbstractions
using CUDA
using Statistics

# Include source files
include("../src/preprocessing.jl")
include("../src/model.jl")
include("../src/fused_loss.jl")

using .Preprocessing
using .RegistrationModel
using .FusedLoss

@testset "Organ Affine Registration Tests" begin

    @testset "Preprocessing" begin
        # Create Dummy Atlas (10x10x10)
        # Organ 1: Center (5,5,5), 3x3x3 block
        atlas_data = zeros(Int, 10, 10, 10)
        atlas_data[4:6, 4:6, 4:6] .= 1

        atlas = MedImage(
            voxel_data = atlas_data,
            origin = (0.0,0.0,0.0),
            spacing = (1.0,1.0,1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.CT_type,
            image_subtype = MedImages.MedImage_data_struct.CT_subtype,
            patient_id = "test_p"
        )

        # Gold same as Atlas
        pts, meta, gold_vol = preprocess_organ_data(atlas, atlas; max_points=10)

        @test size(pts) == (3, 10, 1) # 3 coords, 10 points (downsampled), 1 organ
        @test length(meta) == 1
        @test meta[1].id == 1
        # Barycenter of 3x3x3 block from 4:6 is 5.0
        @test isapprox(meta[1].barycenter[1], 5.0f0, atol=0.1)

        # Check padding
        # Total points in 3x3x3 = 27. Max 10. Should be full.
        @test pts[1, 1, 1] > 0

        # Case with < max_points
        pts_small, _, _ = preprocess_organ_data(atlas, atlas; max_points=100)
        # 27 points, remaining 73 should be -1
        @test pts_small[1, 28, 1] == -1.0f0
    end

    @testset "Model Output Shape" begin
        # Input: (16, 16, 16, 2, 1)
        x = rand(Float32, 16, 16, 16, 2, 1)
        model = MultiScaleCNN(2, 3) # 3 organs

        ps, st = Lux.setup(Random.default_rng(), model)
        y, _ = model(x, ps, st)

        # Output: (15 params, 3 organs, 1 batch)
        @test size(y) == (15, 3, 1)
    end

    @testset "Loss Kernel Forward (Single Batch, Single Organ)" begin
        # 1 Organ, 1 Batch
        points = fill(-1.0f0, (3, 512, 1))
        points[1,1,1] = 5.0f0
        points[2,1,1] = 5.0f0
        points[3,1,1] = 5.0f0

        # Params: Identity
        # Rot=0, Trans=0, Scale=0 (exp(0)=1), Shear=0, Center=0
        params = zeros(Float32, 15, 1, 1)
        # Set scale to 1.0 explicitly since kernel logic multiplies
        # The model output has activation, but here we pass raw params to kernel.
        # Wait, the kernel logic takes raw params.
        # Kernel logic: px *= sx. So if we pass 0, scale is 0!
        # The Model output applies constraints (exp).
        # But this test calls compute_organ_loss directly.
        # So we must pass params that result in Identity behavior.
        # Scale indices 7-9. Set to 1.0.
        params[7:9, 1, 1] .= 1.0f0

        # Gold Vol: 10x10x10x1
        gold = zeros(Float32, 10, 10, 10, 1)
        # Set value at 5,5,5 to 1.0 (perfect match)
        gold[5, 5, 5, 1] = 1.0f0

        # Meta: Center at 5,5,5, Radius 1.0
        meta = [OrganMetadata(1, (5.0f0, 5.0f0, 5.0f0), 1.0f0)]

        loss = compute_organ_loss(points, params, gold, meta)

        # Point (5,5,5) -> Transformed (5,5,5)
        # Dist to barycenter (5,5,5) is 0. 0 < Radius 1.0. Diff = -1.0.
        # Loss1 = log(1 + exp(-1)) ≈ 0.31326
        # Interpolation at 5,5,5 is 1.0. Loss2 = (1-1)^2 = 0.
        # Total = 0.31326

        @test isapprox(loss, log(1.0f0 + exp(-1.0f0)), atol=1e-4)
    end

    @testset "Multi-Batch Multi-Organ Logic" begin
        # 2 Organs, 2 Batches
        points = fill(-1.0f0, (3, 512, 2))
        # Organ 1 Point: (5,5,5)
        points[:, 1, 1] .= 5.0f0
        # Organ 2 Point: (2,2,2)
        points[:, 1, 2] .= 2.0f0

        params = zeros(Float32, 15, 2, 2)
        params[7:9, :, :] .= 1.0f0 # Scale 1

        gold = zeros(Float32, 10, 10, 10, 2) # 2 organs
        gold[5, 5, 5, 1] = 1.0f0
        gold[2, 2, 2, 2] = 1.0f0

        meta = [
            OrganMetadata(1, (5.0f0, 5.0f0, 5.0f0), 1.0f0),
            OrganMetadata(2, (2.0f0, 2.0f0, 2.0f0), 1.0f0)
        ]

        loss = compute_organ_loss(points, params, gold, meta)

        # Both organs should have perfect interpolation (Loss2=0) and Metric 1 ≈ 0.313
        # Average should be ≈ 0.313
        @test isapprox(loss, log(1.0f0 + exp(-1.0f0)), atol=1e-4)
    end

    @testset "Edge Case: Points Out of Bounds" begin
        points = fill(-1.0f0, (3, 512, 1))
        points[:, 1, 1] .= 20.0f0 # Way out of 10x10x10

        params = zeros(Float32, 15, 1, 1)
        params[7:9, :, :] .= 1.0f0

        gold = zeros(Float32, 10, 10, 10, 1)
        meta = [OrganMetadata(1, (5.0f0, 5.0f0, 5.0f0), 1.0f0)]

        loss = compute_organ_loss(points, params, gold, meta)

        # Metric 2 (Interp): Should be 0.0 (extrap). Loss2 = (1-0)^2 = 1.0.
        # Metric 1: Dist(20, 5) ≈ 25.98. Diff = 24.98. Softplus(24.98) ≈ 24.98.
        # Total ≈ 25.98

        dist = sqrt(3 * (15.0f0)^2)
        expected_l1 = log(1.0f0 + exp(dist - 1.0f0))
        expected_l2 = 1.0f0

        @test isapprox(loss, expected_l1 + expected_l2, atol=1e-2)
    end

    @testset "Loss Kernel Gradients" begin
        using Zygote

        points = fill(-1.0f0, (3, 512, 1))
        points[1,1,1] = 5.0f0
        points[2,1,1] = 5.0f0
        points[3,1,1] = 5.0f0

        params = zeros(Float32, 15, 1, 1)
        # Scale = 1.0
        params[7:9, 1, 1] .= 1.0f0
        # Shift translation X to 1.0
        params[4, 1, 1] = 1.0f0

        gold = zeros(Float32, 10, 10, 10, 1)
        gold[6, 5, 5, 1] = 1.0f0 # Target is at 6 (5+1)

        meta = [OrganMetadata(1, (6.0f0, 5.0f0, 5.0f0), 2.0f0)]

        # Calculate gradient w.r.t params
        gs = gradient(p -> compute_organ_loss(points, p, gold, meta), params)[1]

        @test gs !== nothing
        @test size(gs) == size(params)

        # Gradient Logic Verification
        # Current pos: (6, 5, 5). Target in gold: 1.0 at (6,5,5).
        # Metric 2 is minimized (perfect match).
        # Metric 1: Barycenter (6,5,5), Radius 2. Dist 0. Diff -2.
        # Everything aligned.

        # Force mismatch
        # Set gold target at 7. Point is at 6.
        gold[6, 5, 5, 1] = 0.0f0
        gold[7, 5, 5, 1] = 1.0f0

        gs_mismatch = gradient(p -> compute_organ_loss(points, p, gold, meta), params)[1]
        # tx is at index 4.
        # We want to increase tx to move point from 6 to 7.
        # Loss decreases as tx increases. Gradient should be negative?
        # Or if we follow gradient descent (p -= lr * grad), and we want p to increase, grad must be negative.
        # Let's just check magnitude is significant.
        @test abs(gs_mismatch[4,1,1]) > 1e-5
    end

end
