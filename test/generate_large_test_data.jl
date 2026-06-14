using MedImages
using Printf

function generate_large_test_data()
    source_path = "test_data/synthethic_small.nii.gz"
    target_path = "test_data/volume-0.nii.gz"
    func_path = "test_data/filtered_func_data.nii.gz"

    if !isfile(source_path)
        println("Error: source file $source_path not found")
        return
    end

    println("Loading source image: $source_path")
    img = load_image(source_path, "CT")
    
    # Resample to 0.15mm isotropic to get a larger volume (213x213x213)
    # This will ensure cropping tests with 165 max index pass.
    println("Resampling to 0.15mm spacing...")
    target_spacing = (0.15, 0.15, 0.15)
    img_large = resample_to_spacing(img, target_spacing, Linear_en)
    
    println("New size: ", size(img_large.voxel_data))
    
    println("Saving as: $target_path")
    create_nii_from_medimage(img_large, target_path)
    
    println("Saving as: $func_path")
    create_nii_from_medimage(img_large, func_path)
    
    println("Test data generation complete.")
end

generate_large_test_data()
