import nibabel as nib
import numpy as np

pat_dir = "/home/user/MedImages.jl/val_outputs_full/FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat48"
data_dir = "/home/user/MedImages.jl/data/dosimetry_data/FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat48"

try:
    ude = nib.load(f"{pat_dir}/ude_improved_full.nii.gz").get_fdata()
    print("ude max:", np.max(ude))
except Exception as e:
    print("ude error:", e)

try:
    approx = nib.load(f"{data_dir}/dosemap_approx.nii.gz").get_fdata()
    print("approx max:", np.max(approx))
except Exception as e:
    print("approx error:", e)

try:
    semi = nib.load(f"{pat_dir}/semi_dose.nii.gz").get_fdata()
    print("semi max:", np.max(semi))
except Exception as e:
    print("semi error:", e)
    
try:
    base = nib.load(f"{pat_dir}/baseline_analytical.nii.gz").get_fdata()
    print("base max:", np.max(base))
except Exception as e:
    print("base error:", e)
