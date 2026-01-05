#!/usr/bin/env julia
using MedImages
using PyCall

println("=== Verifying Axis Mapping Fixes ===\n")

# Load test image
test_file = "MedImages.jl/test_data/volume-0.nii.gz"
med_im = MedImages.load_image(test_file, "CT")

# Load with SimpleITK
sitk = pyimport("SimpleITK")
sitk_image = sitk.ReadImage(test_file)

println("Original image size:")
println("  Julia: ", size(med_im.voxel_data))
println("  SITK:  ", sitk_image.GetSize())

# Test 1: Pad operation
println("\n=== Test 1: Pad Operation ===")
pad_beg = (10, 11, 13)
pad_end = (10, 11, 13)
pad_val = 0.0

# SimpleITK
extract = sitk.ConstantPadImageFilter()
extract.SetConstant(pad_val)
extract.SetPadLowerBound((UInt(pad_beg[1]), UInt(pad_beg[2]), UInt(pad_beg[3])))
extract.SetPadUpperBound((UInt(pad_end[1]), UInt(pad_end[2]), UInt(pad_end[3])))
sitk_padded = extract.Execute(sitk_image)

# MedImages
med_padded = MedImages.pad_mi(med_im, pad_beg, pad_end, pad_val, MedImages.Linear_en)

println("Padded sizes:")
println("  SITK result:     ", sitk_padded.GetSize())
println("  MedImages result:", size(med_padded.voxel_data))

# Compare voxels
arr = sitk.GetArrayFromImage(sitk_padded)
vox = permutedims(med_padded.voxel_data, (3, 2, 1))

if size(arr) == size(vox)
    max_diff = maximum(abs.(arr .- vox))
    println("  ✓ Dimensions match!")
    println("  Max voxel difference: $max_diff")
    if max_diff < 1e-6
        println("  ✓ Voxels match perfectly!")
    else
        println("  ✗ Voxels differ by $max_diff")
        exit(1)
    end
else
    println("  ✗ Dimension mismatch!")
    println("    SITK:     ", size(arr))
    println("    MedImages:", size(vox))
    exit(1)
end

# Test 2: Crop operation
println("\n=== Test 2: Crop Operation ===")
crop_beg = (0, 0, 0)
crop_size = (151, 156, 50)

# SimpleITK
sitk_cropped = sitk.RegionOfInterest(sitk_image,
    (UInt(crop_size[1]), UInt(crop_size[2]), UInt(crop_size[3])),
    (UInt(crop_beg[1]), UInt(crop_beg[2]), UInt(crop_beg[3])))

# MedImages
med_cropped = MedImages.crop_mi(med_im, crop_beg, crop_size, MedImages.Linear_en)

println("Cropped sizes:")
println("  SITK result:     ", sitk_cropped.GetSize())
println("  MedImages result:", size(med_cropped.voxel_data))

# Compare voxels
arr2 = sitk.GetArrayFromImage(sitk_cropped)
vox2 = permutedims(med_cropped.voxel_data, (3, 2, 1))

if size(arr2) == size(vox2)
    max_diff2 = maximum(abs.(arr2 .- vox2))
    println("  ✓ Dimensions match!")
    println("  Max voxel difference: $max_diff2")
    if max_diff2 < 1e-6
        println("  ✓ Voxels match perfectly!")
    else
        println("  ✗ Voxels differ by $max_diff2")
        exit(1)
    end
else
    println("  ✗ Dimension mismatch!")
    println("    SITK:     ", size(arr2))
    println("    MedImages:", size(vox2))
    exit(1)
end

println("\n=== All Tests Passed! ===")
println("The axis mapping fixes are working correctly.")
exit(0)
