using Test
using MedImages
using MedImages.SUV_calc
using MedImages.MedImage_data_struct
using Dates

@testset "SUV Calculation Tests" begin

    # Mock metadata for a valid PET scan
    # Weight: 75 kg -> 75000 g
    # Dose: 370 MBq -> 3.7e8 Bq
    # Half-life (F-18): 6586.2 s
    # Injection Time: 10:00:00
    # Scan Time: 11:00:00 (Delta = 3600 s)

    valid_meta = Dict{Any, Any}(
        "PatientWeight" => 75.0,
        "RadiopharmaceuticalInformationSequence" => [
            Dict{Any, Any}(
                "RadionuclideTotalDose" => 3.7e8,
                "RadionuclideHalfLife" => 6586.2,
                "RadiopharmaceuticalStartTime" => "100000.00"
            )
        ],
        "AcquisitionTime" => "110000.00"
    )

    # Expected calculation:
    # Delta t = 3600 s
    # Decay factor = 2^(-3600 / 6586.2) ≈ 2^(-0.5466) ≈ 0.6846
    # Actual dose = 3.7e8 * 0.6846 ≈ 2.533e8
    # SUV factor = 75000 / 2.533e8 ≈ 2.96e-4

    # Let's compute exact expected value
    weight_g = 75.0 * 1000.0
    inj_dose = 3.7e8
    half_life = 6586.2
    delta_t = 3600.0
    decay = 2.0^(-delta_t / half_life)
    actual_dose = inj_dose * decay
    expected_factor = weight_g / actual_dose

    @testset "Single MedImage SUV" begin
        # Create a dummy MedImage with this metadata
        # We need to fill required fields for constructor, though SUV only uses metadata
        mi = MedImage(
            voxel_data = zeros(10, 10, 10),
            origin = (0.0, 0.0, 0.0),
            spacing = (1.0, 1.0, 1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.PET_type,
            image_subtype = MedImages.MedImage_data_struct.FDG_subtype,
            patient_id = "test_pat",
            metadata = valid_meta
        )

        factor = MedImages.calculate_suv_factor(mi)
        @test factor !== nothing
        @test isapprox(factor, expected_factor, rtol=1e-5)
    end

    @testset "BatchedMedImage SUV" begin
        # Create a batch of 2 images
        # 1. Valid meta (same as above)
        # 2. Invalid meta (missing weight)

        invalid_meta = deepcopy(valid_meta)
        delete!(invalid_meta, "PatientWeight")

        batch_meta = [valid_meta, invalid_meta]

        bmi = BatchedMedImage(
            voxel_data = zeros(10, 10, 10, 2),
            origin = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
            spacing = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
            direction = [(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)],
            image_type = [MedImages.MedImage_data_struct.PET_type, MedImages.MedImage_data_struct.PET_type],
            image_subtype = [MedImages.MedImage_data_struct.FDG_subtype, MedImages.MedImage_data_struct.FDG_subtype],
            date_of_saving = [Dates.now(), Dates.now()],
            acquistion_time = [Dates.now(), Dates.now()],
            patient_id = ["p1", "p2"],
            metadata = batch_meta
        )

        factors = MedImages.calculate_suv_factor(bmi)

        @test length(factors) == 2
        @test factors[1] !== nothing
        @test isapprox(factors[1], expected_factor, rtol=1e-5)
        @test factors[2] === nothing
    end

    @testset "Missing Metadata Scenarios" begin
        base_mi = MedImage(
            voxel_data = zeros(1,1,1),
            origin = (0.0,0.0,0.0),
            spacing = (1.0,1.0,1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.PET_type,
            image_subtype = MedImages.MedImage_data_struct.FDG_subtype,
            patient_id = "test",
            metadata = Dict{Any, Any}()
        )

        # Case 1: Empty metadata
        @test MedImages.calculate_suv_factor(base_mi) === nothing

        # Case 2: Missing Radiopharmaceutical Sequence
        meta_no_radio = deepcopy(valid_meta)
        delete!(meta_no_radio, "RadiopharmaceuticalInformationSequence")
        base_mi.metadata = meta_no_radio
        @test MedImages.calculate_suv_factor(base_mi) === nothing

        # Case 3: Missing Time
        meta_no_time = deepcopy(valid_meta)
        delete!(meta_no_time, "AcquisitionTime")
        base_mi.metadata = meta_no_time
        @test MedImages.calculate_suv_factor(base_mi) === nothing
    end

    @testset "Time Parsing and Crossover" begin
        # Case: Scan next day (Midnight crossover)
        # Inj: 23:00:00, Scan: 01:00:00 (next day) -> Delta = 2h = 7200s

        meta_cross = deepcopy(valid_meta)
        meta_cross["RadiopharmaceuticalInformationSequence"][1]["RadiopharmaceuticalStartTime"] = "230000.00"
        meta_cross["AcquisitionTime"] = "010000.00"

        delta_t_cross = 7200.0 # 2 hours
        decay_cross = 2.0^(-delta_t_cross / 6586.2)
        actual_dose_cross = 3.7e8 * decay_cross
        expected_cross = (75.0 * 1000.0) / actual_dose_cross

        mi_cross = MedImage(
            voxel_data = zeros(1,1,1),
            origin = (0.0,0.0,0.0),
            spacing = (1.0,1.0,1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.PET_type,
            image_subtype = MedImages.MedImage_data_struct.FDG_subtype,
            patient_id = "test",
            metadata = meta_cross
        )

        factor = MedImages.calculate_suv_factor(mi_cross)
        @test factor !== nothing
        @test isapprox(factor, expected_cross, rtol=1e-5)
    end
end
