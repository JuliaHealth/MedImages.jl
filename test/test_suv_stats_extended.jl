using Test
using MedImages
using MedImages.MedImage_data_struct
using Dates

@testset "Extended SUV Statistics Tests" begin

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

    @testset "Mask with Different Values (Binary Logic)" begin
        img = create_suv_image((10, 10, 10), 70.0, 370e6, 6586.2, "110000", "120000")
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
        # Set some voxels to 1.0, others to 2.0, others to 0.5
        mask.voxel_data[1, 1, 1] = 1.0
        mask.voxel_data[2, 2, 2] = 2.0
        mask.voxel_data[3, 3, 3] = 0.5

        # Total masked voxels = 3

        stats = calculate_suv_statistics(img, mask)
        factor = calculate_suv_factor(img)
        expected_total = 3 * factor # All voxels in image are 1.0
        expected_mean = factor

        @test isapprox(stats.total_suv, expected_total)
        @test isapprox(stats.mean_suv, expected_mean)
    end

    @testset "Edge Cases: Integer Metadata" begin
        # Weight as Int, Dose as Int
        img = create_suv_image((5,5,5), 70, 370000000, 6586.2, "110000", "120000")

        # Should not error
        stats = calculate_suv_statistics(img, img) # Self mask -> all true
        @test stats.mean_suv > 0
    end

    @testset "Edge Cases: Time Formats" begin
        # Test with HHMMSS.ffffff
        img = create_suv_image((5,5,5), 70.0, 370e6, 6586.2, "110000.000000", "120000.500000")

        stats = calculate_suv_statistics(img, img)
        @test stats.mean_suv > 0

        # Check calculation manually: Delta = 3600.5 s
        # Decay = 2^(-3600.5 / 6586.2)
        weight_g = 70.0 * 1000
        dose = 370e6
        decay = 2.0^(-3600.5 / 6586.2)
        expected = weight_g / (dose * decay)

        @test isapprox(stats.mean_suv, expected, rtol=1e-5)
    end

    @testset "Batch Dimension Mismatch (N vs M)" begin
        # Batch size 2
        voxel_data_batch = ones(10, 10, 10, 2)
        img = create_suv_image((10, 10, 10), 70.0, 370e6, 6586.2, "110000", "120000")

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
            metadata = [img.metadata, img.metadata]
        )

        # Batch mask size 3
        mask_batch_data = zeros(10, 10, 10, 3)
        batched_mask_wrong = BatchedMedImage(
            voxel_data = mask_batch_data,
            origin = [img.origin, img.origin, img.origin],
            spacing = [img.spacing, img.spacing, img.spacing],
            direction = [img.direction, img.direction, img.direction],
            image_type = [img.image_type, img.image_type, img.image_type],
            image_subtype = [img.image_subtype, img.image_subtype, img.image_subtype],
            date_of_saving = [Dates.now(), Dates.now(), Dates.now()],
            acquistion_time = [Dates.now(), Dates.now(), Dates.now()],
            patient_id = ["m1", "m2", "m3"],
            metadata = [Dict(), Dict(), Dict()]
        )

        @test_throws DimensionMismatch calculate_suv_statistics(batched_img, batched_mask_wrong)
    end

    @testset "Batched Image (4D) with Single Mask (3D)" begin
         # Batch size 2
        voxel_data_batch = ones(10, 10, 10, 2)
        # Set second batch to 2.0
        voxel_data_batch[:,:,:,2] .= 2.0

        img = create_suv_image((10, 10, 10), 70.0, 370e6, 6586.2, "110000", "120000")

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
            metadata = [img.metadata, img.metadata]
        )

        # Single 3D mask
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
        mask.voxel_data[1:5, 1:5, 1:5] .= 1 # 125 voxels

        stats = calculate_suv_statistics(batched_img, mask)

        factor = calculate_suv_factor(img)

        @test length(stats) == 2
        # First batch has ones
        @test isapprox(stats[1].total_suv, 125 * 1.0 * factor)
        # Second batch has twos
        @test isapprox(stats[2].total_suv, 125 * 2.0 * factor)
    end
end
