using Test
using Lux
using Random
using MedImages
using MedImages.MedImage_data_struct
using KernelAbstractions
using CUDA

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

    @testset "Loss Kernel Forward" begin
        # 1 Organ, 1 Batch
        points = fill(-1.0f0, (3, 512, 1))
        points[1,1,1] = 5.0f0
        points[2,1,1] = 5.0f0
        points[3,1,1] = 5.0f0

        # Params: Identity
        # Rot=0, Trans=0, Scale=0 (exp(0)=1), Shear=0, Center=0
        params = zeros(Float32, 15, 1, 1)

        # Gold Vol: 10x10x10x1
        gold = zeros(Float32, 10, 10, 10, 1)
        # Set value at 5,5,5 to 1.0 (perfect match)
        gold[5, 5, 5, 1] = 1.0f0

        # Meta: Center at 5,5,5, Radius 1.0
        meta = [OrganMetadata(1, (5.0f0, 5.0f0, 5.0f0), 1.0f0)]

        loss = compute_organ_loss(points, params, gold, meta)

        # Point (5,5,5) -> Transformed (5,5,5)
        # Dist to barycenter (5,5,5) is 0. 0 < Radius 1.0. Diff = -1.0.
        # Loss1 = log(1 + exp(-1)) ≈ 0.313
        # Interpolation at 5,5,5 is 1.0. Loss2 = (1-1)^2 = 0.
        # Total = 0.313

        @test isapprox(loss, log(1.0f0 + exp(-1.0f0)), atol=1e-4)
    end

    @testset "Loss Kernel Gradients" begin
        using Zygote

        points = fill(-1.0f0, (3, 512, 1))
        points[1,1,1] = 5.0f0
        points[2,1,1] = 5.0f0
        points[3,1,1] = 5.0f0

        params = zeros(Float32, 15, 1, 1)
        # Shift translation X to 1.0
        params[4, 1, 1] = 1.0f0

        gold = zeros(Float32, 10, 10, 10, 1)
        gold[6, 5, 5, 1] = 1.0f0 # Target is at 6 (5+1)

        meta = [OrganMetadata(1, (6.0f0, 5.0f0, 5.0f0), 2.0f0)]

        # Calculate gradient w.r.t params
        gs = gradient(p -> compute_organ_loss(points, p, gold, meta), params)[1]

        @test gs !== nothing
        @test size(gs) == size(params)
        # Should have gradient on tx (index 4)
        # If we move tx, loss changes (Metric 2 stays low, Metric 1 changes distance)
        # Actually Metric 2 is perfect (6 matches 6). Metric 1: dist(6,6)=0.
        # Let's create a mismatch to force gradient.
        # Set gold at 7. Point moves to 6.
        # Interpolation at 6 is 0. Loss2 is high.
        # Gradient should pull tx towards +1 (to reach 7).

        gold[6, 5, 5, 1] = 0.0f0
        gold[7, 5, 5, 1] = 1.0f0

        gs_mismatch = gradient(p -> compute_organ_loss(points, p, gold, meta), params)[1]
        # tx is at 4.
        # Point is at 5+1 = 6. Target is 7.
        # Gradient should indicate direction.
        @test abs(gs_mismatch[4,1,1]) > 0.0f0
    end

end
