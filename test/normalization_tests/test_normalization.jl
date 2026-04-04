using Test
using MedImages
using Statistics
using Dates
using LinearAlgebra

@testset "MedImages.Normalization Tests" begin

    # Mock MedImage
    voxel_data = rand(Float32, 10, 10, 10) .* 100.0
    mi = MedImage(
        voxel_data = voxel_data,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = Tuple(collect(Iterators.flatten(I(3)))),
        image_type = MedImages.MRI_type,
        image_subtype = MedImages.T1_subtype,
        patient_id = "test_patient",
        metadata = Dict("RescaleSlope" => 2.0, "RescaleIntercept" => 10.0)
    )

    @testset "Z-score Normalization" begin
        normalized_mi = z_score_normalize(mi)
        @test mean(normalized_mi.voxel_data) ≈ 0.0 atol=1e-5
        @test std(normalized_mi.voxel_data) ≈ 1.0 atol=1e-5
        @test size(normalized_mi.voxel_data) == size(mi.voxel_data)
    end

    @testset "Min-Max Normalization" begin
        normalized_mi = min_max_normalize(mi, range=(0.0, 1.0))
        @test minimum(normalized_mi.voxel_data) ≈ 0.0 atol=1e-5
        @test maximum(normalized_mi.voxel_data) ≈ 1.0 atol=1e-5
    end

    @testset "DICOM Rescale" begin
        rescaled_mi = apply_dicom_rescale(mi)
        expected = mi.voxel_data .* 2.0 .+ 10.0
        @test all(rescaled_mi.voxel_data .≈ expected)
    end

    @testset "Histogram Matching" begin
        target_data = rand(Float32, 10, 10, 10) .+ 50.0 # Shifted distribution
        target_mi = update_voxel_data(mi, target_data)
        
        matched_mi = histogram_match(mi, target_mi)
        
        # Checking if distributions are similar
        @test mean(matched_mi.voxel_data) ≈ mean(target_data) atol=1.0
        @test std(matched_mi.voxel_data) ≈ std(target_data) atol=1.0
    end

    @testset "Nyul Normalization" begin
        # Create a "cohort"
        images = [mi, deepcopy(mi)]
        images[2].voxel_data .+= 10.0
        
        train_results = nyul_train(images)
        transformed_mi = nyul_transform(mi, train_results)
        
        @test size(transformed_mi.voxel_data) == size(mi.voxel_data)
        # Landmaks should match train results roughly
        @test quantile(reshape(transformed_mi.voxel_data, :), 0.5) ≈ train_results[2][findfirst(x->x==50, train_results[1])] atol=1.0
    end

    @testset "BatchedMedImage Normalization" begin
        batch_data = rand(Float32, 5, 5, 5, 2)
        bmi = create_batched_medimage([mi, mi])
        bmi.voxel_data = batch_data
        
        normalized_bmi = z_score_normalize(bmi)
        for i in 1:2
            vol = normalized_bmi.voxel_data[:, :, :, i]
            @test mean(vol) ≈ 0.0 atol=1e-5
            @test std(vol) ≈ 1.0 atol=1e-5
        end
        
        min_max_bmi = min_max_normalize(bmi, range=(0.0, 1.0))
        for i in 1:2
            vol = min_max_bmi.voxel_data[:, :, :, i]
            @test minimum(vol) ≈ 0.0 atol=1e-5
            @test maximum(vol) ≈ 1.0 atol=1e-5
        end
    end

end
