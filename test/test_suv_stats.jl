using Test
using MedImages
using MedImages.MedImage_data_struct
using Dates
using Accessors


@testset "SUV Statistics Tests" begin

    # Helper to create dummy MedImage with SUV metadata
    function create_suv_image(shape, weight, dose, half_life, inj_time, scan_time)
        meta = Dict{Any, Any}()
        meta["PatientWeight"] = weight
        meta["AcquisitionTime"] = scan_time

        radio_info = Dict{Any, Any}()
        radio_info["RadionuclideTotalDose"] = dose
        radio_info["RadionuclideHalfLife"] = half_life
        radio_info["RadiopharmaceuticalStartTime"] = inj_time

        meta["RadiopharmaceuticalInformationSequence"] = [radio_info]

        return MedImage(
            voxel_data = ones(shape...),
            origin = (0.0, 0.0, 0.0),
            spacing = (1.0, 1.0, 1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.PET_type,
            image_subtype = MedImages.MedImage_data_struct.FDG_subtype,
            patient_id = "test_patient",
            metadata = meta
        )
    end

    @testset "Single Image + Single Mask" begin
        img = create_suv_image((10, 10, 10), 70.0, 370e6, 6586.2, "110000", "120000")
        mask = MedImage(
            voxel_data = zeros(10, 10, 10), # All zeros
            origin = (0.0, 0.0, 0.0),
            spacing = (1.0, 1.0, 1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.PET_type,
            image_subtype = MedImages.MedImage_data_struct.FDG_subtype,
            patient_id = "mask",
            metadata = Dict()
        )

        # Test empty mask
        stats = calculate_suv_statistics(img, mask)
        @test stats.mean_suv == 0.0
        @test stats.total_suv == 0.0

        # Set some voxels to 1
        mask.voxel_data[1:5, 1:5, 1:5] .= 1
        # Count should be 125
        stats = calculate_suv_statistics(img, mask)

        # Calculate expected
        factor = calculate_suv_factor(img)
        expected_total = 125 * factor # Since all pixels are 1
        expected_mean = factor

        @test isapprox(stats.total_suv, expected_total)
        @test isapprox(stats.mean_suv, expected_mean)

        # Test Dimension Mismatch
        mask_wrong = MedImage(
            voxel_data = zeros(5, 5, 5),
            origin = (0.0, 0.0, 0.0),
            spacing = (1.0, 1.0, 1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.PET_type,
            image_subtype = MedImages.MedImage_data_struct.FDG_subtype,
            patient_id = "mask_wrong",
            metadata = Dict()
        )
        @test_throws DimensionMismatch calculate_suv_statistics(img, mask_wrong)

        # Test Missing Metadata
        img_bad = deepcopy(img)
        delete!(img_bad.metadata, "PatientWeight")
        @test_throws ErrorException calculate_suv_statistics(img_bad, mask)
    end

    @testset "Batched Image + Single Mask" begin
        # Create batched image (2 batches)
        voxel_data_batch = ones(10, 10, 10, 2)
        voxel_data_batch[:,:,:,2] .= 2.0

        # Base image for metadata
        img = create_suv_image((10, 10, 10), 70.0, 370e6, 6586.2, "110000", "120000")

        meta1 = img.metadata
        meta2 = deepcopy(meta1)
        meta2["PatientWeight"] = 140.0

        batched_img = BatchedMedImage(
            voxel_data = voxel_data_batch,
            origin = [img.origin, img.origin],
            spacing = [img.spacing, img.spacing],
            direction = [img.direction, img.direction],
            image_type = [img.image_type, img.image_type],
            image_subtype = [img.image_subtype, img.image_subtype],
            date_of_saving = [Dates.now(), Dates.now()],
            acquistion_time = [Dates.now(), Dates.now()],
            patient_id = ["p1", "p2"],
            metadata = [meta1, meta2]
        )

        mask = MedImage(
            voxel_data = zeros(10, 10, 10),
            origin = (0.0, 0.0, 0.0),
            spacing = (1.0, 1.0, 1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.PET_type,
            image_subtype = MedImages.MedImage_data_struct.FDG_subtype,
            patient_id = "mask",
            metadata = Dict()
        )
        mask.voxel_data[1:5, 1:5, 1:5] .= 1

        stats_batch = calculate_suv_statistics(batched_img, mask)

        factor1 = calculate_suv_factor(img)
        img2_mock = @set img.metadata = meta2;
        factor2 = calculate_suv_factor(img2_mock)

        @test length(stats_batch) == 2
        @test isapprox(stats_batch[1].total_suv, 125 * factor1)
        @test isapprox(stats_batch[2].total_suv, 125 * 2.0 * factor2)

        # Test Dimension Mismatch (Batch Slice)
        mask_wrong = MedImage(
            voxel_data = zeros(5, 5, 5),
            origin = (0.0, 0.0, 0.0),
            spacing = (1.0, 1.0, 1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.PET_type,
            image_subtype = MedImages.MedImage_data_struct.FDG_subtype,
            patient_id = "mask",
            metadata = Dict()
        )
        @test_throws DimensionMismatch calculate_suv_statistics(batched_img, mask_wrong)
    end

    @testset "Batched Image + Batched Mask" begin
        # Create batched image (2 batches)
        voxel_data_batch = ones(10, 10, 10, 2)
        voxel_data_batch[:,:,:,2] .= 2.0

        img = create_suv_image((10, 10, 10), 70.0, 370e6, 6586.2, "110000", "120000")
        meta1 = img.metadata
        meta2 = deepcopy(meta1)
        meta2["PatientWeight"] = 140.0

        batched_img = BatchedMedImage(
            voxel_data = voxel_data_batch,
            origin = [img.origin, img.origin],
            spacing = [img.spacing, img.spacing],
            direction = [img.direction, img.direction],
            image_type = [img.image_type, img.image_type],
            image_subtype = [img.image_subtype, img.image_subtype],
            date_of_saving = [Dates.now(), Dates.now()],
            acquistion_time = [Dates.now(), Dates.now()],
            patient_id = ["p1", "p2"],
            metadata = [meta1, meta2]
        )

        mask_batch_data = zeros(10, 10, 10, 2)
        mask_batch_data[1:5, 1:5, 1:5, 1] .= 1 # 125 voxels
        mask_batch_data[1:2, 1:2, 1:2, 2] .= 1 # 8 voxels

        batched_mask = BatchedMedImage(
            voxel_data = mask_batch_data,
            origin = [img.origin, img.origin],
            spacing = [img.spacing, img.spacing],
            direction = [img.direction, img.direction],
            image_type = [img.image_type, img.image_type],
            image_subtype = [img.image_subtype, img.image_subtype],
            date_of_saving = [Dates.now(), Dates.now()],
            acquistion_time = [Dates.now(), Dates.now()],
            patient_id = ["m1", "m2"],
            metadata = [Dict(), Dict()]
        )

        stats_batch_mask = calculate_suv_statistics(batched_img, batched_mask)

        factor1 = calculate_suv_factor(img)
        img2_mock = @set img.metadata = meta2;
        factor2 = calculate_suv_factor(img2_mock)

        @test length(stats_batch_mask) == 2
        @test isapprox(stats_batch_mask[1].total_suv, 125 * factor1)
        @test isapprox(stats_batch_mask[2].total_suv, 8 * 2.0 * factor2)

        # Test Dimension Mismatch
        mask_wrong_data = zeros(10, 10, 10, 3) # 3 batches
        batched_mask_wrong = @set batched_mask.voxel_data = mask_wrong_data

        @test_throws DimensionMismatch calculate_suv_statistics(batched_img, batched_mask_wrong)
    end
end
