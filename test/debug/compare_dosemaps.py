import nibabel as nib
import numpy as np
import sys

pat_dir = "/DATA/FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_2__Pat54"
mc_path = f"{pat_dir}/DOSEMAP/dosemap.nii.gz"
approx_path = f"{pat_dir}/SPECT_DATA/nifti_files/Dosemap.nii.gz"

mc_nii = nib.load(mc_path)
approx_nii = nib.load(approx_path)

print("Monte Carlo (DOSEMAP/dosemap.nii.gz):")
print(f"  Shape: {mc_nii.shape}")
print(f"  Pixdim: {mc_nii.header['pixdim'][1:4]}")
mc_data = mc_nii.get_fdata()
print(f"  Sum: {np.sum(mc_data)}")
print(f"  Max: {np.max(mc_data)}")
print(f"  Mean: {np.mean(mc_data)}")

print("\nApproximate (SPECT_DATA/nifti_files/Dosemap.nii.gz):")
print(f"  Shape: {approx_nii.shape}")
print(f"  Pixdim: {approx_nii.header['pixdim'][1:4]}")
approx_data = approx_nii.get_fdata()

print(f"  Sum: {np.sum(approx_data)}")
print(f"  Max: {np.max(approx_data)}")
print(f"  Mean: {np.mean(approx_data)}")

if mc_data.shape == approx_data.shape:
    try:
        corr = np.corrcoef(mc_data.flatten(), approx_data.flatten())[0,1]
        print(f"\nCorrelation: {corr}")
    except:
        print("\nCould not calculate correlation.")
else:
    print(f"\nShapes do not match, cannot compute direct correlation without resampling.")
