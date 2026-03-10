# test/test_rotation_non_iso.jl

using Test
using MedImages
using LinearAlgebra

@testset "Non-isovolumetric Rotation Verification" begin
    # Create a synthetic non-isovolumetric image
    # Size: (10, 10, 20)
    # Spacing: (1.0, 1.0, 2.0)
    # This means the Z-axis is "stretched" compared to X and Y in voxel space
    data = zeros(Float32, 10, 10, 20)
    # Put a "marker" block around (5, 5, 10) in voxel space so it survives linear interpolation
    data[4:6, 4:6, 9:11] .= 1.0

    
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 2.0)
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    
    im = MedImage(
        voxel_data = data,
        origin = origin,
        spacing = spacing,
        direction = direction,
        image_type = MedImages.MRI_type,
        image_subtype = MedImages.T1_subtype,
        patient_id = "test_non_iso"
    )
    
    # Rotate 90 degrees around Y-axis (axis=2)
    # In physical space, a point (x, y, z) rotates to (z, y, -x) around origin
    # For our marker (5, 5, 20) -> (20, 5, -5)
    # But rotate_mi rotates around the CENTER of the image.
    
    angle = 90.0 # degrees
    axis = 2
    
    @info "Rotating non-isovolumetric image..."
    im_rotated = rotate_mi(im, axis, angle, MedImages.Linear_en)
    
    @test size(im_rotated.voxel_data) == size(im.voxel_data)
    @test im_rotated.spacing == im.spacing
    
    # Check if the rotated image still has data (basic check)
    @test sum(im_rotated.voxel_data) > 0
    
    # Check isovolumetric case for consistency
    data_iso = zeros(Float32, 10, 10, 10)
    data_iso[4:6, 4:6, 4:6] .= 1.0

    im_iso = MedImage(
        voxel_data = data_iso,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = direction,
        image_type = MedImages.MRI_type,
        image_subtype = MedImages.T1_subtype,
        patient_id = "test_iso"
    )
    
    @info "Rotating isovolumetric image..."
    im_iso_rotated = rotate_mi(im_iso, axis, angle, MedImages.Linear_en)
    @test sum(im_iso_rotated.voxel_data) > 0
    
    @info "Verification complete."
end
