push!(LOAD_PATH, joinpath(pwd(), "src"))
using MedImages
using MedImages.Load_and_save
using MedImages.MedImage_data_struct
using MedImages.Spatial_metadata_change
using MedImages.Resample_to_target
using MedImages.Utils
using CUDA
using Test
using Dates

function verify_integration()
    println("--- Verifying Library Integration of Fused Kernels ---")
    
    # 1. Setup mock data
    # Create uniform sized images for batching
    data1 = rand(Float32, 64, 64, 64)
    data2 = rand(Float32, 64, 64, 64)
    
    # Standard direction (identity matrix for RAS)
    identity_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    
    im1 = MedImage(
        voxel_data = data1,
        spacing = (1.0, 1.0, 1.0),
        origin = (0.0, 0.0, 0.0),
        direction = identity_direction,
        image_type = MedImage_data_struct.CT_type,
        image_subtype = MedImage_data_struct.CT_subtype,
        patient_id = "test1"
    )
    
    im2 = MedImage(
        voxel_data = data2,
        spacing = (1.0, 1.0, 1.0),
        origin = (5.0, 5.0, 5.0),
        direction = identity_direction,
        image_type = MedImage_data_struct.PET_type,
        image_subtype = MedImage_data_struct.FDG_subtype,
        patient_id = "test2"
    )
    
    # 2. Test resample_to_spacing (Batched)
    println("Testing resample_to_spacing (Batched)...")
    
    batch_im = create_batched_medimage([im1, im2])
    
    target_spacing = (2.0, 2.0, 2.0)
    
    # Execute
    resampled_batch = resample_to_spacing(batch_im, target_spacing, MedImage_data_struct.Linear_en)
    
    @test size(resampled_batch.voxel_data, 4) == 2
    @test resampled_batch.spacing[1] == target_spacing
    @test resampled_batch.spacing[2] == target_spacing
    
    println("  Success: resample_to_spacing produced $(size(resampled_batch.voxel_data)[1:3]) from $(size(batch_im.voxel_data)[1:3])")

    # 3. Test resample_to_image (Batched)
    println("Testing resample_to_image (Batched)...")
    
    # Create fixed image batch using create_batched_medimage for consistency
    f1 = MedImage(
        voxel_data = zeros(Float32, 50, 50, 50),
        spacing = (1.5, 1.5, 1.5),
        origin = (10.0, 10.0, 10.0),
        direction = identity_direction,
        image_type = MedImage_data_struct.CT_type,
        image_subtype = MedImage_data_struct.CT_subtype,
        patient_id = "fixed1"
    )
    f2 = MedImage(
        voxel_data = zeros(Float32, 50, 50, 50),
        spacing = (1.5, 1.5, 1.5),
        origin = (-5.0, -5.0, -5.0),
        direction = identity_direction,
        image_type = MedImage_data_struct.CT_type,
        image_subtype = MedImage_data_struct.CT_subtype,
        patient_id = "fixed2"
    )
    
    fixed_batch = create_batched_medimage([f1, f2])
    
    # Move to GPU if needed
    if Utils.is_cuda_array(batch_im.voxel_data)
        fixed_batch.voxel_data = CuArray(fixed_batch.voxel_data)
    end
    
    # Resample moving batch (batch_im) to fixed_batch
    resampled_to_img = resample_to_image(fixed_batch, batch_im, MedImage_data_struct.Linear_en)
    
    @test size(resampled_to_img.voxel_data)[1:3] == (50, 50, 50)
    @test resampled_to_img.origin[1] == (10.0, 10.0, 10.0)
    @test resampled_to_img.origin[2] == (-5.0, -5.0, -5.0)
    
    println("  Success: resample_to_image produced (50,50,50) batch.")

    # 4. Correctness Check (Single vs Batched)
    println("Correctness check (Single vs Batched)...")
    
    # Single resample for item 2
    im2_resampled_single = resample_to_image(unbatch_medimage(fixed_batch)[2], im2, MedImage_data_struct.Linear_en)
    
    # Batched resample item 2
    im2_resampled_batched = unbatch_medimage(resampled_to_img)[2]
    
    # Compare voxel data (some tolerance due to Float32)
    diff = sum(abs.(im2_resampled_single.voxel_data .- im2_resampled_batched.voxel_data)) / prod(size(im2_resampled_single.voxel_data))
    println("  Mean absolute diff: $diff")
    @test diff < 1e-4
    
    println("--- Verification Complete ---")
end

if CUDA.functional()
    verify_integration()
else
    println("CUDA not available, skipping verification.")
end
