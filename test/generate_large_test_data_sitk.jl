using PyCall

function generate_large_test_data_sitk()
    sitk = pyimport("SimpleITK")
    source_path = "test_data/synthethic_small.nii.gz"
    target_path = "test_data/volume-0.nii.gz"
    func_path = "test_data/filtered_func_data.nii.gz"

    if !isfile(source_path)
        println("Error: source file $source_path not found")
        return
    end

    println("Loading source image via SimpleITK: $source_path")
    img = sitk.ReadImage(source_path)
    
    # Resample to 0.15mm isotropic
    println("Resampling to 0.15mm spacing via SimpleITK...")
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    
    target_spacing = (0.15, 0.15, 0.15)
    new_size = [UInt32(round(old_size[i] * old_spacing[i] / target_spacing[i])) for i in 1:3]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    
    img_large = resampler.Execute(img)
    
    println("New size: ", img_large.GetSize())
    
    println("Saving as: $target_path")
    sitk.WriteImage(img_large, target_path)
    
    println("Saving as: $func_path")
    sitk.WriteImage(img_large, func_path)
    
    println("Test data generation complete.")
end

generate_large_test_data_sitk()
