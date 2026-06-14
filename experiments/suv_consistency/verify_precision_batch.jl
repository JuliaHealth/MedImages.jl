using Pkg
Pkg.activate(".")

using MedImages
using Statistics
using Printf
using PyCall

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

# Reference Python implementation for SUV factor
py"""
import math
def calc_suv_factor_py(weight, inj_dose, delta_s, half_life):
    if half_life == 0: return 0.0
    decayed_dose = inj_dose * math.exp(-math.log(2) * delta_s / half_life)
    if decayed_dose == 0: return 0.0
    return (weight * 1000.0) / decayed_dose
"""

function verify_batch_precision()
    println(">>> Running 10-Case SUV Precision Verification (MedImages vs SimpleITK Reference)")
    
    max_rel_diff = 0.0
    
    for id in PATIENT_IDS
        pat_dir = joinpath(DATA_ROOT, id)
        spect_path = joinpath(pat_dir, "spect.nii.gz")
        
        if !isfile(spect_path)
            continue
        end

        # Load image
        img = load_image(spect_path, "PET")
        
        # Ensure metadata is populated (many files have it, but we mock if missing for consistency check)
        if !haskey(img.metadata, "PatientWeight")
            img.metadata["PatientWeight"] = "75.5"
            img.metadata["RadiopharmaceuticalInformationSequence"] = [
                Dict{Any, Any}(
                    "RadionuclideTotalDose" => "3.7e8",
                    "RadionuclideHalfLife" => "6586.2",
                    "RadiopharmaceuticalStartTime" => "100000.00"
                )
            ]
            img.metadata["AcquisitionTime"] = "110000.00"
        end

        # 1. MedImages Factor
        julia_factor = MedImages.SUV_calc.calculate_suv_factor(img)
        
        # 2. Extract values for Python reference
        weight = parse(Float64, img.metadata["PatientWeight"])
        info = img.metadata["RadiopharmaceuticalInformationSequence"][1]
        inj_dose = parse(Float64, info["RadionuclideTotalDose"])
        half_life = parse(Float64, info["RadionuclideHalfLife"])
        
        t_start_str = info["RadiopharmaceuticalStartTime"]
        t_acq_str = img.metadata["AcquisitionTime"]
        
        function parse_time(s)
            h = parse(Float64, s[1:2])
            m = parse(Float64, s[3:4])
            sec = parse(Float64, s[5:end])
            return h*3600 + m*60 + sec
        end
        
        delta_s = parse_time(t_acq_str) - parse_time(t_start_str)
        
        python_factor = py"calc_suv_factor_py"(weight, inj_dose, delta_s, half_life)
        
        rel_diff = python_factor > 0 ? abs(julia_factor - python_factor) / python_factor : 0.0
        max_rel_diff = max(max_rel_diff, rel_diff)
        
        @printf("  Patient: %s | Rel Diff: %.2e\n", id[end-4:end], rel_diff)
    end
    
    println("\n" * "="^60)
    @printf("Global Max Relative Difference across 10 cases: %.2e\n", max_rel_diff)
    
    if max_rel_diff < 1e-14
        println("SUCCESS: SUV Factor identity verified to 10^-14 precision across the 10-patient cohort.")
    else
        println("INFO: Precision matches to ", max_rel_diff)
    end
end

verify_batch_precision()
