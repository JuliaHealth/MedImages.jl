using Dates
using MedImages
using Zygote
using Test
using CUDA

function test_suv_differentiability()
    println("Testing SUV differentiability...")
    
    # Create a mock MedImage
    voxel_data = rand(Float32, 10, 10, 10)
    meta = Dict{Any, Any}(
        "PatientWeight" => 70.0,
        "RadiopharmaceuticalInformationSequence" => [
            Dict(
                "RadionuclideTotalDose" => 3.7e8,
                "RadionuclideHalfLife" => 6586.2,
                "RadiopharmaceuticalStartTime" => "100000.000000"
            )
        ],
        "AcquisitionTime" => "110000.000000"
    )
    
    mi = MedImage(
        voxel_data = voxel_data,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        image_type = MedImages.PET_type,
        image_subtype = MedImages.FDG_subtype,
        patient_id = "test_patient",
        metadata = meta
    )
    
    # Define a loss function that depends on voxel data after SUV application
    loss(data) = sum(apply_suv(MedImage(
        voxel_data = data,
        origin = mi.origin,
        spacing = mi.spacing,
        direction = mi.direction,
        image_type = mi.image_type,
        image_subtype = mi.image_subtype,
        patient_id = mi.patient_id,
        date_of_saving = mi.date_of_saving,
        acquistion_time = mi.acquistion_time,
        metadata = mi.metadata
    )).voxel_data)
    
    # Test gradient
    grad = Zygote.gradient(loss, voxel_data)[1]
    
    factor = calculate_suv_factor(mi)
    @test all(grad .≈ Float32(factor))
    println("Single image differentiability: PASSED (Factor: $factor)")
end

function test_suv_batched_differentiability()
    println("Testing Batched SUV differentiability...")
    
    B = 2
    voxel_data = rand(Float32, 10, 10, 10, B)
    meta = [
        Dict{Any, Any}(
            "PatientWeight" => 70.0,
            "RadiopharmaceuticalInformationSequence" => [
                Dict("RadionuclideTotalDose" => 3.7e8, "RadionuclideHalfLife" => 6586.2, "RadiopharmaceuticalStartTime" => "100000.000000")
            ],
            "AcquisitionTime" => "110000.000000"
        ),
        Dict{Any, Any}(
            "PatientWeight" => 80.0,
            "RadiopharmaceuticalInformationSequence" => [
                Dict("RadionuclideTotalDose" => 3.7e8, "RadionuclideHalfLife" => 6586.2, "RadiopharmaceuticalStartTime" => "100000.000000")
            ],
            "AcquisitionTime" => "110000.000000"
        )
    ]
    
    bmi = BatchedMedImage(
        voxel_data = voxel_data,
        origin = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        spacing = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
        direction = [(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)],
        image_type = [MedImages.PET_type, MedImages.PET_type],
        image_subtype = [MedImages.FDG_subtype, MedImages.FDG_subtype],
        patient_id = ["p1", "p2"],
        date_of_saving = [Dates.today(), Dates.today()],
        acquistion_time = [Dates.now(), Dates.now()],
        metadata = meta
    )
    
    loss(data) = sum(apply_suv(BatchedMedImage(
        voxel_data = data,
        origin = bmi.origin,
        spacing = bmi.spacing,
        direction = bmi.direction,
        image_type = bmi.image_type,
        image_subtype = bmi.image_subtype,
        patient_id = bmi.patient_id,
        date_of_saving = bmi.date_of_saving,
        acquistion_time = bmi.acquistion_time,
        metadata = bmi.metadata
    )).voxel_data)
    
    grad = Zygote.gradient(loss, voxel_data)[1]
    
    factors = calculate_suv_factor(bmi)
    @test all(grad[:,:,:,1] .≈ Float32(factors[1]))
    @test all(grad[:,:,:,2] .≈ Float32(factors[2]))
    println("Batched differentiability: PASSED (Factors: $factors)")
end

function test_suv_gpu()
    if !CUDA.functional()
        println("CUDA not available, skipping GPU test.")
        return
    end
    
    println("Testing SUV on GPU...")
    voxel_data = cu(rand(Float32, 10, 10, 10))
    meta = Dict{Any, Any}(
        "PatientWeight" => 70.0,
        "RadiopharmaceuticalInformationSequence" => [
            Dict(
                "RadionuclideTotalDose" => 3.7e8,
                "RadionuclideHalfLife" => 6586.2,
                "RadiopharmaceuticalStartTime" => "100000.000000"
            )
        ],
        "AcquisitionTime" => "110000.000000"
    )
    
    mi = MedImage(
        voxel_data = voxel_data,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        image_type = MedImages.PET_type,
        image_subtype = MedImages.FDG_subtype,
        patient_id = "test_gpu",
        metadata = meta
    )
    
    res = apply_suv(mi)
    @test res.voxel_data isa CuArray
    println("GPU SUV Application: PASSED")
end

@testset "Differentiable SUV Tests" begin
    test_suv_differentiability()
    test_suv_batched_differentiability()
    test_suv_gpu()
end
