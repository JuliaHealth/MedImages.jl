using Test
using MedImages
using Zygote
using FiniteDifferences
using LinearAlgebra
using Dates
using MedImages.MedImage_data_struct

const TEST_MRI = MedImages.MedImage_data_struct.MRI_type
const TEST_SUBTYPE = MedImages.MedImage_data_struct.T1_subtype
const TEST_LINEAR = MedImages.MedImage_data_struct.Linear_en
const TEST_LPI = MedImages.MedImage_data_struct.ORIENTATION_LPI

# Fixed datetime values to avoid Zygote trying to differentiate through Dates.now()
const FIXED_DATE = DateTime(2024, 1, 1)

# Helper to create mock image
function create_mock_medimage(data)
    MedImage(
        voxel_data = data,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        image_type = TEST_MRI,
        image_subtype = TEST_SUBTYPE,
        patient_id = "test_patient",
        date_of_saving = FIXED_DATE,
        acquistion_time = FIXED_DATE
    )
end

@testset "Differentiability Tests" begin

    @testset "resample_to_image" begin
        # Moving image data
        data_moving = rand(Float32, 10, 10, 10)
        # Fixed image (defines grid)
        data_fixed = zeros(Float32, 5, 5, 5) # Smaller grid
        im_fixed = create_mock_medimage(data_fixed)
        im_fixed.spacing = (2.0, 2.0, 2.0) # Different spacing

        function loss(x)
            # Create fresh moving image with tracked data
            im_mov = create_mock_medimage(x)
            # Resample
            resampled = MedImages.resample_to_image(im_fixed, im_mov, TEST_LINEAR)
            return sum(resampled.voxel_data)
        end

        # Test gradient
        grads = Zygote.gradient(loss, data_moving)
        @test grads[1] !== nothing
        @test !all(iszero, grads[1])
    end

    @testset "rotate_mi" begin
        data = rand(Float32, 10, 10, 10)

        function loss(x)
            im = create_mock_medimage(x)
            # Rotate 45 deg around axis 1
            rotated = MedImages.rotate_mi(im, 1, 45.0, TEST_LINEAR)
            return sum(rotated.voxel_data)
        end

        grads = Zygote.gradient(loss, data)
        # Verify gradient existence (might fail if warp is not differentiable)
        @test grads[1] !== nothing
    end

    @testset "scale_mi" begin
        data = rand(Float32, 10, 10, 10)

        function loss(x)
            im = create_mock_medimage(x)
            scaled = MedImages.scale_mi(im, 0.5, TEST_LINEAR)
            return sum(scaled.voxel_data)
        end

        grads = Zygote.gradient(loss, data)
        @test grads[1] !== nothing
    end

    @testset "translate_mi" begin
        data = rand(Float32, 10, 10, 10)

        function loss(x)
            im = create_mock_medimage(x)
            translated = MedImages.translate_mi(im, 5, 1, TEST_LINEAR)
            return sum(translated.voxel_data)
        end

        grads = Zygote.gradient(loss, data)
        @test grads[1] !== nothing
        # Gradient should be all ones because it's identity on pixels
        @test all(isapprox.(grads[1], 1.0))
    end

    @testset "crop_mi" begin
        data = rand(Float32, 10, 10, 10)

        function loss(x)
            im = create_mock_medimage(x)
            cropped = MedImages.crop_mi(im, (2,2,2), (4,4,4), TEST_LINEAR)
            return sum(cropped.voxel_data)
        end

        grads = Zygote.gradient(loss, data)
        @test grads[1] !== nothing
        # Gradient should be 1.0 inside crop, 0.0 outside
        @test isapprox(grads[1][4,4,4], 1.0)
        @test isapprox(grads[1][1,1,1], 0.0)
    end

    @testset "pad_mi" begin
        data = rand(Float32, 5, 5, 5)

        function loss(x)
            im = create_mock_medimage(x)
            padded = MedImages.pad_mi(im, (2,2,2), (2,2,2), 0.0, TEST_LINEAR)
            return sum(padded.voxel_data)
        end

        grads = Zygote.gradient(loss, data)
        @test grads[1] !== nothing
        @test all(isapprox.(grads[1], 1.0))
    end

     @testset "resample_to_spacing" begin
        data = rand(Float32, 10, 10, 10)

        function loss(x)
            im = create_mock_medimage(x)
            # Change spacing to larger
            resampled = MedImages.resample_to_spacing(im, (2.0, 2.0, 2.0), TEST_LINEAR)
            return sum(resampled.voxel_data)
        end

        grads = Zygote.gradient(loss, data)
        @test grads[1] !== nothing
        @test !all(iszero, grads[1])
    end

    @testset "change_orientation" begin
        data = rand(Float32, 5, 5, 5)

        function loss(x)
            im = create_mock_medimage(x)
            # Change orientation
            reoriented = MedImages.change_orientation(im, TEST_LPI)
            return sum(reoriented.voxel_data)
        end

        grads = Zygote.gradient(loss, data)
        @test grads[1] !== nothing
        # Should be permuted/reversed ones?
        @test all(isapprox.(grads[1], 1.0))
    end

end
