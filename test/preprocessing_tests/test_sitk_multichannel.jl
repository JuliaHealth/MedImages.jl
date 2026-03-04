# test/preprocessing_tests/test_sitk_multichannel.jl
using Test
using MedImages
using PyCall
using CUDA

sitk = pyimport("SimpleITK")

@testset "SimpleITK Multichannel Resampling Verification" begin
    # 1. Create a dummy test image (small size for speed)
    dims = (32, 32, 32)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    
    # Use deterministic data
    data1 = collect(reshape(range(0.0f0, stop=1.0f0, length=prod(dims)), dims))
    data2 = 1.0f0 .- data1
    data_4d = cat(data1, data2; dims=4)
    
    im_4d = MedImage(
        voxel_data = data_4d,
        origin = origin,
        spacing = spacing,
        direction = direction,
        image_type = MedImages.MedImage_data_struct.CT_type,
        image_subtype = MedImages.MedImage_data_struct.CT_subtype,
        patient_id = "test_sitk"
    )
    
    new_spacing = (1.5, 1.5, 1.5)
    
    # 2. Resample using MedImages
    resampled_mi = resample_to_spacing(im_4d, new_spacing, Linear_en)
    
    # 3. Resample using SimpleITK (channel by channel)
    resampled_sitk_channels = []
    # In MedImages we use (X, Y, Z). sitk.GetImageFromArray expects (Z, Y, X).
    # MedImages spacing is (x,y,z). sitk.SetSpacing expects (x,y,z).
    
    new_size_calc = Tuple{Int,Int,Int}(ceil.((dims .* spacing) ./ new_spacing))

    for c in 1:2
        channel_data = data_4d[:, :, :, c]
        # permute to (Z, Y, X) for SimpleITK
        sitk_img = sitk.GetImageFromArray(permutedims(channel_data, (3, 2, 1)))
        sitk_img.SetSpacing(spacing)
        sitk_img.SetOrigin(origin)
        sitk_img.SetDirection(direction)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize((UInt32(new_size_calc[1]), UInt32(new_size_calc[2]), UInt32(new_size_calc[3])))
        resampler.SetOutputDirection(direction)
        resampler.SetOutputOrigin(origin)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        res_sitk = resampler.Execute(sitk_img)
        res_arr = sitk.GetArrayFromImage(res_sitk) # PyCall converts to Array
        
        # permute back to (X, Y, Z)
        push!(resampled_sitk_channels, permutedims(res_arr, (3, 2, 1)))
    end
    
    resampled_sitk_4d = cat(resampled_sitk_channels...; dims=4)
    
    # 4. Compare
    @test size(resampled_mi.voxel_data) == size(resampled_sitk_4d)
    @test resampled_mi.spacing == new_spacing
    
    # Tolerance for interpolation (SimpleITK vs MedImages kernel)
    max_diff = maximum(abs.(resampled_mi.voxel_data .- resampled_sitk_4d))
    @info "Max difference between MedImages and SimpleITK: $max_diff"
    @test all(isapprox.(resampled_mi.voxel_data, resampled_sitk_4d, atol=1e-4))
    
    # 5. Verify BatchedMedImage resampling (integer spacing)
    # Simple BatchedMedImage test (no channels here, Batch is 4th dim)
    b_data = cat(data1, data1 .+ 0.5f0; dims=4) # 32x32x32x2
    b_im = create_batched_medimage([
        MedImage(voxel_data=data1, origin=origin, spacing=spacing, direction=direction, patient_id="p1", image_type=MedImages.MedImage_data_struct.CT_type, image_subtype=MedImages.MedImage_data_struct.CT_subtype),
        MedImage(voxel_data=data1 .+ 0.5f0, origin=origin, spacing=spacing, direction=direction, patient_id="p2", image_type=MedImages.MedImage_data_struct.CT_type, image_subtype=MedImages.MedImage_data_struct.CT_subtype)
    ])
    
    resampled_b_mi = resample_to_spacing(b_im, new_spacing, Linear_en)
    @test size(resampled_b_mi.voxel_data) == (new_size_calc..., 2)
    @test resampled_b_mi.spacing[1] == new_spacing
    
    # Compare first item with single resample
    single1 = resample_to_spacing(MedImage(voxel_data=data1, origin=origin, spacing=spacing, direction=direction, patient_id="p1", image_type=MedImages.MedImage_data_struct.CT_type, image_subtype=MedImages.MedImage_data_struct.CT_subtype), new_spacing, Linear_en)
    @test all(isapprox.(resampled_b_mi.voxel_data[:,:,:,1], single1.voxel_data, atol=1e-4))
end
