using MedImages
using Statistics
using Printf

# Function to calculate volume in cm^3
function calculate_volume_cm3(mask::MedImage)
    voxel_count = sum(mask.voxel_data .> 0.5)
    voxel_volume_mm3 = prod(mask.spacing)
    return (voxel_count * voxel_volume_mm3) / 1000.0
end

# Helper to print stats
function print_stats(label, pet, mask)
    # calculate_suv_statistics is exported by MedImages
    stats = calculate_suv_statistics(pet, mask)
    vol = calculate_volume_cm3(mask)
    @printf("%-35s | Mean SUV: %8.4f | Volume (cm3): %8.2f\n", label, stats.mean_suv, vol)
    return stats.mean_suv, vol
end

function run_experiment()
    # 1. Load data
    # Note: Using exact matching for filenames with double spaces
    patient_id = "fdg_0168f65af8_04-04-2003-NA-PET-CT Ganzkoerper  primaer mit KM-82224"
    pet_path = "test_data/autoPET-III-Lite/Images-PET/$(patient_id)_0001.nii.gz"
    ct_path = "test_data/autoPET-III-Lite/Images-CT/$(patient_id)_0000.nii.gz"
    mask_path = "test_data/autoPET-III-Lite/Masks/$(patient_id).nii.gz"

    println("Loading patient: $patient_id")
    if !isfile(pet_path)
        error("PET file not found: $pet_path")
    end

    pet = load_image(pet_path, "PET")
    ct = load_image(ct_path, "CT")
    mask = load_image(mask_path, "PET") # Mask is aligned with PET usually

    # Mock metadata for autoPET, which is already in SUV. 
    # Create metadata that results in a 1.0 conversion factor.
    # formula: (weight_kg * 1000) / decayed_dose = 1.0 
    # => 100kg * 1000 / 100000 = 1.0
    pet.metadata["PatientWeight"] = "100.0"
    pet.metadata["RadiopharmaceuticalInformationSequence"] = [
        Dict{Any, Any}(
            "RadionuclideTotalDose" => "100000.0",
            "RadionuclideHalfLife" => "6586.2",
            "RadiopharmaceuticalStartTime" => "120000.000000"
        )
    ]
    pet.metadata["AcquisitionTime"] = "120000.000000"

    # 2. Initial baseline
    println("\n" * "="^90)
    println("Baseline (Original Space)")
    m0, v0 = print_stats("Original", pet, mask)
    
    # Save the 'before' state
    mkdir("experiments/suv_consistency/before")
    create_nii_from_medimage(pet, "experiments/suv_consistency/before/bef_pet.nii.gz")
    create_nii_from_medimage(ct, "experiments/suv_consistency/before/bef_ct.nii.gz")
    create_nii_from_medimage(mask, "experiments/suv_consistency/before/bef_mask.nii.gz")

    # 3. Resample PET and Mask to CT space
    println("\nStep 1: Resample to CT target...")
    pet_ct = resample_to_image(ct, pet, Linear_en)
    mask_ct = resample_to_image(ct, mask, Nearest_neighbour_en)
    m1, v1 = print_stats("After Resample to CT", pet_ct, mask_ct)

    # 4. Change spacing to 2.0mm isotropic
    println("\nStep 2: Change spacing to 2.0mm isotropic...")
    new_spacing = (2.0, 2.0, 2.0)
    pet_sp = resample_to_spacing(pet_ct, new_spacing, Linear_en)
    mask_sp = resample_to_spacing(mask_ct, new_spacing, Nearest_neighbour_en)
    m2, v2 = print_stats("After Spacing Change", pet_sp, mask_sp)

    # 5. Rotate 45 degrees around Z axis (axis 3)
    println("\nStep 3: Rotate 45 degrees around Z...")
    pet_rot = rotate_mi(pet_sp, 3, 45.0, Linear_en, true)
    mask_rot = rotate_mi(mask_sp, 3, 45.0, Nearest_neighbour_en, true)
    m3, v3 = print_stats("After Rotation", pet_rot, mask_rot)

    # 6. Translate (shift origin)
    println("\nStep 4: Translate (Origin shift along X by 10 voxels)...")
    # translate_mi(im, translate_by, axis, interpolator)
    pet_final = translate_mi(pet_rot, 10, 1, Linear_en)
    mask_final = translate_mi(mask_rot, 10, 1, Nearest_neighbour_en)
    m4, v4 = print_stats("Final (After Translate)", pet_final, mask_final)

    # Save the 'after' state
    # We will resample the CT to the final space so they match visually.
    # No, wait, CT was just a target. We can just apply the same steps to CT, 
    # but the prompt just said "save files before and after tranformations label PET CT SUV"
    # Actually wait, maybe I should also transform CT so that we can visualize it together!
    ct_sp = resample_to_spacing(ct, new_spacing, Linear_en)
    ct_rot = rotate_mi(ct_sp, 3, 45.0, Linear_en, true)
    ct_final = translate_mi(ct_rot, 10, 1, Linear_en)

    mkdir("experiments/suv_consistency/after")
    create_nii_from_medimage(pet_final, "experiments/suv_consistency/after/aft_pet.nii.gz")
    create_nii_from_medimage(ct_final, "experiments/suv_consistency/after/aft_ct.nii.gz")
    create_nii_from_medimage(mask_final, "experiments/suv_consistency/after/aft_mask.nii.gz")

    println("\n" * "="^90)
    println("Consistency Summary (Deltas Final vs Original):")
    @printf("Mean SUV Delta:   %10.6f (%.4f%%)\n", abs(m4 - m0), abs(m4 - m0)/m0 * 100)
    @printf("Volume Delta:     %10.6f (%.4f%%)\n", abs(v4 - v0), abs(v4 - v0)/v0 * 100)
    
    # We compare against original directly (m0, v0)
    if abs(m4 - m0)/m0 < 0.05 && abs(v4 - v0)/v0 < 0.05
        println("\nSUCCESS: SUV and Volume are consistent within 5% error margin.")
        println("Note: Small variations are expected due to interpolation and grid sampling.")
    else
        println("\nWARNING: Large discrepancy detected. Check interpolation and metadata preservation.")
    end
end

run_experiment()
