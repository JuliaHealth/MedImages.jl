using MedImages
using MedImages.MedImage_data_struct
using MedImages.Basic_transformations
using Statistics
using LinearAlgebra

println("--- DEBUG BATCHED ROTATION ---")
size_t = (32, 32, 32)
data = zeros(Float32, size_t)
data[10:20, 10:15, 10:20] .= 1.0

img = MedImage(
    voxel_data=data, 
    origin=(0.0,0.0,0.0), 
    spacing=(1.0,1.0,1.0), 
    direction=(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0), 
    image_type=MedImage_data_struct.MRI_type, 
    image_subtype=MedImage_data_struct.T1_subtype,
    patient_id="p1"
)
batch = create_batched_medimage([img, img])

angles = [0.0, 90.0]
# We want to see what happens inside rotate_mi(batch, 3, angles, Linear_en)
batch_size = 2
matrices = map(1:batch_size) do b
    current_angle = angles[b]
    current_direction = batch.direction[b]
    R_sub = Rodrigues_rotation_matrix(current_direction, 3, current_angle)
    println("Matrix $b (angle $(angles[b])): ", R_sub)
    return [R_sub[1,1] R_sub[1,2] R_sub[1,3] 0.0;
            R_sub[2,1] R_sub[2,2] R_sub[2,3] 0.0;
            R_sub[3,1] R_sub[3,2] R_sub[3,3] 0.0;
            0.0        0.0        0.0        1.0]
end

rot_batch = affine_transform_mi(batch, matrices, Linear_en)

diff_0 = abs.(rot_batch.voxel_data[:,:,:,1] - batch.voxel_data[:,:,:,1])
m_diff = mean(diff_0)
max_diff = maximum(diff_0)
println("0 deg mean diff: $m_diff")
println("0 deg max diff: $max_diff")

if m_diff >= 1e-4
    println("FAILURE: Mean diff $m_diff >= 1e-4")
else
    println("SUCCESS: Mean diff $m_diff < 1e-4")
end
