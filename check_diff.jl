using Pkg; Pkg.activate("/workspaces/MedImages.jl")
using MedImages
using MedImages.Resample_to_target
using MedImages.MedImage_data_struct
using PyCall

sitk = pyimport("SimpleITK")
im_fixed = load_image("/workspaces/MedImages.jl/test_data/volume-0.nii.gz", "CT")
im_moving = load_image("/workspaces/MedImages.jl/test_data/synthethic_small.nii.gz", "CT")
im_resampled = Resample_to_target.resample_to_image(im_fixed, im_moving, MedImages.Linear_en, Float32)

sitk_fixed = sitk.ReadImage("/workspaces/MedImages.jl/test_data/volume-0.nii.gz")
sitk_moving = sitk.ReadImage("/workspaces/MedImages.jl/test_data/synthethic_small.nii.gz")
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(sitk_fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0.0)
sitk_resampled = resampler.Execute(sitk_moving)

sitk_arr = sitk.GetArrayFromImage(sitk_resampled)
sitk_arr = permutedims(sitk_arr, (3, 2, 1))

diff = abs.(sitk_arr .- im_resampled.voxel_data)
println("Max diff: ", maximum(diff))
println("Mean diff: ", sum(diff)/length(diff))
println("Number of voxels with diff > 1.0: ", count(diff .> 1.0))
