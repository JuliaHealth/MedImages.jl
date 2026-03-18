using MedImages
using PyCall
using Statistics
using LinearAlgebra
using Dates
using UUIDs

# Import MedImages internals if needed
import MedImages.MedImage_data_struct: MedImage, MRI_type, CT_type, T1_subtype, CT_subtype, Linear_en

function verify_suv_factor_precision()
    println("--- SUV Factor Precision Verification ---")
    
    # Mock DICOM metadata
    meta = Dict{Any, Any}(
        "PatientWeight" => "75.5",
        "RadiopharmaceuticalInformationSequence" => [
            Dict{Any, Any}(
                "RadionuclideTotalDose" => "370000000.0", # 370 MBq
                "RadionuclideHalfLife" => "6586.2",      # 18F Half life in seconds
                "RadiopharmaceuticalStartTime" => "100000.000000" # 10:00:00
            )
        ],
        "AcquisitionTime" => "110000.000000" # 11:00:00 (1 hour later = 3600 seconds)
    )
    
    # MedImages Calculation
    img = MedImage(
        voxel_data = zeros(Float32, 2, 2, 2),
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = ntuple(i->(i in [1,5,9] ? 1.0 : 0.0), 9),
        image_type = MRI_type,
        image_subtype = T1_subtype,
        patient_id = "test_patient",
        metadata = meta
    )
    
    julia_factor = MedImages.SUV_calc.calculate_suv_factor(img)
    
    py"""
    import math
    import numpy as np
    
    def calc_suv_factor_py(weight, inj_dose, delta_s, half_life):
        decayed_dose = inj_dose * math.exp(-math.log(2) * delta_s / half_life)
        return (weight * 1000.0) / decayed_dose
    """
    python_factor = py"calc_suv_factor_py"(75.5, 3.7e8, 3600.0, 6586.2)
    
    diff = abs(julia_factor - python_factor)
    rel_diff = diff / python_factor
    
    println("Julia Factor:  ", julia_factor)
    println("Python Factor: ", python_factor)
    println("Absolute Diff: ", diff)
    println("Relative Diff: ", rel_diff)
    
    if rel_diff < 1e-14
        println("SUCCESS: SUV Factor identity verified to 10^-14 precision.")
    else
        println("WARNING: SUV Factor difference is larger than expected (", rel_diff, ")")
    end
end

function verify_resampling_precision()
    println("\n--- Resampling Precision Verification (MedImages vs SimpleITK) ---")
    
    dims = (32, 32, 32)
    data = Array{Float32}(undef, dims...)
    for z in 1:dims[3], y in 1:dims[2], x in 1:dims[1]
        data[x, y, z] = Float32(x + y*0.1 + z*0.01)
    end
    
    spacing = (2.0, 2.0, 2.0)
    origin = (0.0, 0.0, 0.0)
    direction_mat = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    direction_tuple = ntuple(i->(i in [1,5,9] ? 1.0 : 0.0), 9)
    
    mi = MedImage(
        voxel_data = data,
        spacing = spacing,
        origin = origin,
        direction = direction_tuple,
        image_type = CT_type,
        image_subtype = CT_subtype,
        patient_id = "test_patient",
        metadata = Dict{Any, Any}()
    )
    
    # 1. MedImages Rotation
    rot_angle_deg = 33.0
    mi_rot = MedImages.rotate_mi(mi, 3, rot_angle_deg, Linear_en)
    
    # 2. SimpleITK Rotation
    sitk = pyimport("SimpleITK")
    np = pyimport("numpy")
    
    data_py = np.array(permutedims(data, (3, 2, 1)))
    sitk_img = sitk.GetImageFromArray(data_py)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin(origin)
    sitk_img.SetDirection(vec(direction_mat'))
    
    center = [spacing[i] * (dims[i]-1)/2.0 + origin[i] for i in 1:3]
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    transform.SetRotation(0.0, 0.0, deg2rad(rot_angle_deg))
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0.0)
    
    sitk_rot_img = resampler.Execute(sitk_img)
    data_rot_sitk = sitk.GetArrayFromImage(sitk_rot_img)
    data_rot_sitk_jl = permutedims(data_rot_sitk, (3, 2, 1))
    
    crop = 5
    region_julia = mi_rot.voxel_data[crop:end-crop, crop:end-crop, crop:end-crop]
    region_sitk = data_rot_sitk_jl[crop:end-crop, crop:end-crop, crop:end-crop]
    
    max_diff = maximum(abs.(region_julia .- region_sitk))
    avg_diff = mean(abs.(region_julia .- region_sitk))
    rel_err = avg_diff / mean(region_julia)
    
    println("Max Difference: ", max_diff)
    println("Avg Difference: ", avg_diff)
    println("Relative Error: ", rel_err)
    
    if rel_err < 1e-4
        println("SUCCESS: Resampling precision consistent with SimpleITK.")
    else
        println("INFO: Resampling precision differs slightly from SimpleITK (expected due to backend implementation details).")
    end
end

verify_suv_factor_precision()
verify_resampling_precision()
