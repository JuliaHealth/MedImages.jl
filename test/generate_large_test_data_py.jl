using PyCall

function generate_large_test_data_py()
    source_path = "test_data/synthethic_small.nii.gz"
    target_path = "test_data/volume-0.nii.gz"
    func_path = "test_data/filtered_func_data.nii.gz"

    if !isfile(source_path)
        println("Error: source file $source_path not found")
        return
    end

    py"""
    import SimpleITK as sitk
    import numpy as np

    def resample_image(source_path, target_path, func_path):
        img = sitk.ReadImage(source_path)
        old_spacing = img.GetSpacing()
        old_size = img.GetSize()
        
        target_spacing = (0.15, 0.15, 0.15)
        new_size = [int(round(old_size[i] * old_spacing[i] / target_spacing[i])) for i in range(3)]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        
        img_large = resampler.Execute(img)
        sitk.WriteImage(img_large, target_path)
        sitk.WriteImage(img_large, func_path)
        return img_large.GetSize()
    """

    println("Resampling via Python/SimpleITK...")
    new_size = py"resample_image"(source_path, target_path, func_path)
    println("New size: ", new_size)
    println("Test data generation complete.")
end

generate_large_test_data_py()
