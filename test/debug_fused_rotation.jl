using Test
using MedImages
using MedImages.MedImage_data_struct
using MedImages.Basic_transformations
using Statistics

@testset "Isolated Batched Rotation Test" begin
    size_t = (32, 32, 32)
    data = zeros(Float32, size_t)
    # Put a block at center
    data[10:20, 10:15, 10:20] .= 1.0

    img = MedImage(voxel_data=data, origin=(0.0,0.0,0.0), spacing=(1.0,1.0,1.0), direction=(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0), image_type=MedImages.MedImage_data_struct.MRI_type, image_subtype=MedImages.MedImage_data_struct.T1_subtype, patient_id="p1")
    batch = create_batched_medimage([img, img])

    # Rotate (Unique)
    # Image 1: 0 deg, Image 2: 90 deg
    angles = [0.0, 90.0]
    rot_batch = rotate_mi(batch, 3, angles, Linear_en)
    
    @test size(rot_batch.voxel_data) == (32, 32, 32, 2)
    
    # 0 deg should be same
    diff_0 = abs.(rot_batch.voxel_data[:,:,:,1] - batch.voxel_data[:,:,:,1])
    mean_diff_0 = mean(diff_0)
    max_diff_0 = maximum(diff_0)
    println("0 deg mean diff: $mean_diff_0")
    println("0 deg max diff: $max_diff_0")
    
    @test mean_diff_0 < 1e-4
    
    # 90 deg should differ
    diff_90 = abs.(rot_batch.voxel_data[:,:,:,2] - batch.voxel_data[:,:,:,2])
    mean_diff_90 = mean(diff_90)
    println("90 deg mean diff: $mean_diff_90")
    @test mean_diff_90 > 0.01
end
