using Test
using MedImages
using MedImages.SUV_calc
using MedImages.MedImage_data_struct
using Dates
using Accessors

@testset "SUV Calculation Tests" begin

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

    weight_g = 75.0 * 1000.0
    inj_dose = 3.7e8
    half_life = 6586.2
    delta_t = 3600.0
    decay = 2.0^(-delta_t / half_life)
    actual_dose = inj_dose * decay
    expected_factor = weight_g / actual_dose

    @testset "Single MedImage SUV" begin
        mi = MedImage(
            voxel_data = zeros(10, 10, 10),
            origin = (0.0, 0.0, 0.0),
            spacing = (1.0, 1.0, 1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = PET_type,
            image_subtype = FDG_subtype,
            patient_id = "test_pat",
            metadata = valid_meta
        )

        factor = calculate_suv_factor(mi)
        @test factor !== nothing
        @test isapprox(factor, expected_factor, rtol=1e-5)
    end

    @testset "BatchedMedImage SUV" begin
        invalid_meta = deepcopy(valid_meta)
        delete!(invalid_meta, "PatientWeight")

        batch_meta = [valid_meta, invalid_meta]

        bmi = BatchedMedImage(
            voxel_data = zeros(10, 10, 10, 2),
            origin = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
            spacing = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
            direction = [(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)],
            image_type = [PET_type, PET_type],
            image_subtype = [FDG_subtype, FDG_subtype],
            date_of_saving = [Dates.now(), Dates.now()],
            acquistion_time = [Dates.now(), Dates.now()],
            patient_id = ["p1", "p2"],
            metadata = batch_meta
        )

        factors = calculate_suv_factor(bmi)

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
            image_type = PET_type,
            image_subtype = FDG_subtype,
            patient_id = "test",
            metadata = Dict{Any, Any}()
        )

        @test calculate_suv_factor(base_mi) === nothing

        meta_no_radio = deepcopy(valid_meta)
        delete!(meta_no_radio, "RadiopharmaceuticalInformationSequence")
        base_mi = @set base_mi.metadata = meta_no_radio
        @test calculate_suv_factor(base_mi) === nothing

        meta_no_time = deepcopy(valid_meta)
        delete!(meta_no_time, "AcquisitionTime")
        base_mi = @set base_mi.metadata = meta_no_time
        @test calculate_suv_factor(base_mi) === nothing
    end

    @testset "Time Parsing and Crossover" begin
        meta_cross = deepcopy(valid_meta)
        meta_cross["RadiopharmaceuticalInformationSequence"][1]["RadiopharmaceuticalStartTime"] = "230000.00"
        meta_cross["AcquisitionTime"] = "010000.00"

        delta_t_cross = 7200.0
        decay_cross = 2.0^(-delta_t_cross / 6586.2)
        actual_dose_cross = 3.7e8 * decay_cross
        expected_cross = (75.0 * 1000.0) / actual_dose_cross

        mi_cross = MedImage(
            voxel_data = zeros(1,1,1),
            origin = (0.0,0.0,0.0),
            spacing = (1.0,1.0,1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = PET_type,
            image_subtype = FDG_subtype,
            patient_id = "test",
            metadata = meta_cross
        )

        factor = calculate_suv_factor(mi_cross)
        @test factor !== nothing
        @test isapprox(factor, expected_cross, rtol=1e-5)
    end
end
