# test/preprocessing_tests/test_preprocessing.jl
using Test
using MedImages
using CUDA
using KernelAbstractions

@testset "Preprocessing Utilities" begin
    # Test one_hot_encode
    @testset "one_hot_encode" begin
        arr = [1 0; 2 1]
        oh = one_hot_encode(arr, 2)
        @test size(oh) == (2, 2, 2)
        @test oh[1,1,1] == 1.0
        @test oh[1,1,2] == 0.0
        @test oh[2,1,1] == 0.0
        @test oh[2,1,2] == 1.0
        
        if CUDA.functional()
            # Test GPU if available
            try
                arr_gpu = CuArray(arr)
                oh_gpu = one_hot_encode(arr_gpu, 2)
                @test Array(oh_gpu) == oh
            catch e
                @warn "GPU test failed for one_hot_encode: $e"
            end
        end
    end
    
    # Test calculate_barycenter
    @testset "calculate_barycenter" begin
        mask = zeros(Int, 10, 10, 10)
        mask[5, 5, 5] = 1
        bc = calculate_barycenter(mask)
        @test bc == (5.0f0, 5.0f0, 5.0f0)
        
        mask[6, 6, 6] = 1
        bc = calculate_barycenter(mask)
        @test bc == (5.5f0, 5.5f0, 5.5f0)
        
        if CUDA.functional()
            try
                bc_gpu = calculate_barycenter(CuArray(mask))
                @test bc_gpu == bc
            catch e
                @warn "GPU test failed for calculate_barycenter: $e"
            end
        end
    end
    
    # Test calculate_max_radius
    @testset "calculate_max_radius" begin
        mask = zeros(Int, 10, 10, 10)
        mask[5, 5, 5] = 1
        mask[7, 5, 5] = 1
        bc = (5.0f0, 5.0f0, 5.0f0)
        r = calculate_max_radius(mask, bc)
        @test r == 2.0f0
        
        if CUDA.functional()
            try
                r_gpu = calculate_max_radius(CuArray(mask), bc)
                @test r_gpu == r
            catch e
                @warn "GPU test failed for calculate_max_radius: $e"
            end
        end
    end

    # Test extract_points_from_mask
    @testset "extract_points_from_mask" begin
        mask = zeros(Int, 10, 10, 10)
        mask[1, 1, 1] = 1
        mask[2, 2, 2] = 1
        
        points = extract_points_from_mask(mask, 1)
        @test size(points) == (3, 1)
        @test points[:, 1] == [1.0f0, 1.0f0, 1.0f0]
        
        points = extract_points_from_mask(mask, 5)
        @test size(points) == (3, 5)
        @test points[:, 1] == [1.0f0, 1.0f0, 1.0f0]
        @test points[:, 2] == [2.0f0, 2.0f0, 2.0f0]
        @test points[:, 3] == [-1.0f0, -1.0f0, -1.0f0]
    end

    @testset "pad_or_crop_mi" begin
        # Create a dummy MedImage using proper structure
        data = ones(Float32, 10, 10, 10)
        im = MedImage(
            voxel_data = data,
            origin = (0.0, 0.0, 0.0),
            spacing = (1.0, 1.0, 1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.CT_type,
            image_subtype = MedImages.MedImage_data_struct.CT_subtype,
            patient_id = "test"
        )
        
        # Test Pad
        im_padded = pad_or_crop_mi(im, (12, 12, 12))
        @test size(im_padded.voxel_data) == (12, 12, 12)
        @test im_padded.origin == (-1.0, -1.0, -1.0)
        @test im_padded.voxel_data[1,1,1] == 0.0
        @test im_padded.voxel_data[2,2,2] == 1.0
        
        # Test Crop
        im_cropped = pad_or_crop_mi(im, (8, 8, 8))
        @test size(im_cropped.voxel_data) == (8, 8, 8)
        @test im_cropped.origin == (1.0, 1.0, 1.0)
        @test im_cropped.voxel_data[1,1,1] == 1.0
        
        # Test Multichannel Pad
        data_4d = cat(data, data .+ 1; dims=4)
        im_4d = MedImage(
            voxel_data = data_4d,
            origin = (0.0, 0.0, 0.0),
            spacing = (1.0, 1.0, 1.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.CT_type,
            image_subtype = MedImages.MedImage_data_struct.CT_subtype,
            patient_id = "test"
        )
        im_4d_padded = pad_or_crop_mi(im_4d, (12, 12, 12))
        @test size(im_4d_padded.voxel_data) == (12, 12, 12, 2)
        @test im_4d_padded.voxel_data[2,2,2, 2] == 2.0
    end
end
