using Pkg
Pkg.activate(".")
# Pkg.instantiate() # Already instantiated

using MedImages
using Statistics
using Printf
using CSV
using DataFrames

# IDs from TRAINING section of splits.txt
const PATIENT_IDS = [
    "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat50",
    "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat58",
    "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat45",
    "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_3__Pat46",
    "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat54",
    "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_6__Pat47",
    "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat55",
    "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat53",
    "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_4__Pat52",
    "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat56"
]

const DATA_ROOT = "data/dosimetry_data"
const MASK_DIR = "experiments/suv_consistency/batch_masks"
const TOTALSEG_BIN = "/home/user/miniforge/envs/sciml_env/bin/TotalSegmentator"

mkpath(MASK_DIR)

function calculate_volume_cm3(mask::MedImage)
    voxel_count = sum(mask.voxel_data .> 0.5)
    voxel_volume_mm3 = prod(mask.spacing)
    return (voxel_count * voxel_volume_mm3) / 1000.0
end

function run_batch_experiment()
    results = DataFrame(
        PatientID = String[],
        Organ = String[],
        MeanSUV_Orig = Float64[],
        MeanSUV_Trans = Float64[],
        SUV_Diff_Perc = Float64[],
        Vol_Orig = Float64[],
        Vol_Trans = Float64[],
        Vol_Diff_Perc = Float64[]
    )

    for id in PATIENT_IDS
        println("\n>>> Processing Patient: $id")
        pat_dir = joinpath(DATA_ROOT, id)
        ct_path = joinpath(pat_dir, "ct.nii.gz")
        spect_path = joinpath(pat_dir, "spect.nii.gz")
        
        if !isfile(ct_path) || !isfile(spect_path)
            println("  [SKIP] Missing CT or SPECT files.")
            continue
        end

        pat_mask_dir = joinpath(MASK_DIR, id)
        mkpath(pat_mask_dir)
        liver_mask_path = joinpath(pat_mask_dir, "liver.nii.gz")
        
        if !isfile(liver_mask_path)
            println("  Running TotalSegmentator for liver mask...")
            cmd = `$TOTALSEG_BIN -i $ct_path -o $pat_mask_dir --task total --fast --roi_subset liver --quiet`
            try run(cmd) catch; end
        end

        if !isfile(liver_mask_path)
            println("  [ERROR] Liver mask not generated for $id")
            continue
        end

        println("  Loading data...")
        spect_raw = load_image(spect_path, "PET")
        ct = load_image(ct_path, "CT")
        mask_raw = load_image(liver_mask_path, "PET") 

        # Ensure metadata for SUV
        for img in [spect_raw]
            if !haskey(img.metadata, "PatientWeight")
                img.metadata["PatientWeight"] = "75.0"
                img.metadata["RadiopharmaceuticalInformationSequence"] = [
                    Dict{Any, Any}(
                        "RadionuclideTotalDose" => "3.7e8",
                        "RadionuclideHalfLife" => "6586.2",
                        "RadiopharmaceuticalStartTime" => "120000.00"
                    )
                ]
                img.metadata["AcquisitionTime"] = "120000.00"
            end
        end

        # --- Baseline Alignment ---
        # Since mask is on CT grid, we align SPECT to CT to define the 'Ground Truth' state for consistency check
        println("  Aligning SPECT to CT for baseline...")
        spect = resample_to_image(ct, spect_raw, Linear_en)
        mask = mask_raw # Already on CT grid

        stats0 = calculate_suv_statistics(spect, mask)
        v0 = calculate_volume_cm3(mask)
        m0 = stats0.mean_suv
        
        if m0 < 1e-6
            @printf("  [WARNING] Mean SUV is near zero (%.2e). Mask might not overlap activity.\n", m0)
        end

        # --- Transformation Pipeline ---
        println("  Applying transformations...")
        # Step 1: 2.0mm isotropic
        new_spacing = (2.0, 2.0, 2.0)
        spect_sp = resample_to_spacing(spect, new_spacing, Linear_en)
        mask_sp = resample_to_spacing(mask, new_spacing, Nearest_neighbour_en)

        # Step 2: Rotate 45 deg Z
        spect_rot = rotate_mi(spect_sp, 3, 45.0, Linear_en, true)
        mask_rot = rotate_mi(mask_sp, 3, 45.0, Nearest_neighbour_en, true)

        # Step 3: Translate 10 voxels X
        spect_final = translate_mi(spect_rot, 10, 1, Linear_en)
        mask_final = translate_mi(mask_rot, 10, 1, Nearest_neighbour_en)

        # Statistics
        stats_f = calculate_suv_statistics(spect_final, mask_final)
        vf = calculate_volume_cm3(mask_final)
        mf = stats_f.mean_suv

        suv_diff = abs(mf - m0) / (m0 + 1e-9) * 100
        vol_diff = abs(vf - v0) / (v0 + 1e-9) * 100
        
        push!(results, (
            id, "Liver", m0, mf, suv_diff, v0, vf, vol_diff
        ))
        
        @printf("  [RESULT] SUV Diff: %.4f%% | Vol Diff: %.4f%%\n", suv_diff, vol_diff)
    end

    CSV.write("experiments/suv_consistency/batch_results_10cases.csv", results)
    println("\n" * "="^60)
    println("Batch Consistency Summary (Mean over $(nrow(results)) cases):")
    # Fixed typo in column name below
    println("Mean SUV Deviation:   ", mean(results.SUV_Diff_Perc), "%")
    println("Mean Volume Deviation: ", mean(results.Vol_Diff_Perc), "%")
end

run_batch_experiment()
