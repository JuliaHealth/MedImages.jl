using Test
using MedImages
using Zygote
using MedImages.MedImage_data_struct: MRI_type, T1_subtype, Linear_en

function create_mock_medimage(data)
    MedImage(
        voxel_data = data,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        image_type = MRI_type,
        image_subtype = T1_subtype,
        patient_id = "test_patient"
    )
end

@testset "pad_mi" begin
    data = rand(Float32, 5, 5, 5)

    function loss(x)
        im = create_mock_medimage(x)
        padded = MedImages.pad_mi(im, (1,1,1), (1,1,1), 0.0, Linear_en)
        return sum(padded.voxel_data)
    end

    grads = Zygote.gradient(loss, data)
    @test grads[1] !== nothing
    @test all(isapprox.(grads[1], 1.0))
end
