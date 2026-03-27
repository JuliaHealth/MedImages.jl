import os
import nibabel as nib
import numpy as np
import torch
from pytheranostics.dosimetry.VoxelSDosimetry import VoxelSDosimetry
from scipy.stats import pearsonr

# Define a simple mock longitudinal study class to satisfy pytheranostics if needed, 
# or use their tools directly if we can figure out the structure.
# Based on Tools.py, it expects NM and CT longitudinal studies.

class MockLongStudy:
    def __init__(self, data, spacing, origin):
        self.images = [data] # List of 3D arrays
        self.meta = [{"spacing": spacing, "origin": origin, "AcquisitionDate": "20260101", "AcquisitionTime": "120000"}]

def generate_vsv_baseline(data_dir, pat_name):
    print(f"Generating VSV Baseline for {pat_name}...")
    pat_dir = os.path.join(data_dir, pat_name)
    
    ct_img = nib.load(os.path.join(pat_dir, "ct.nii.gz"))
    spect_img = nib.load(os.path.join(pat_dir, "spect.nii.gz"))
    mc_img = nib.load(os.path.join(pat_dir, "dosemap_mc.nii.gz"))
    
    ct_data = ct_img.get_fdata()
    spect_data = spect_img.get_fdata()
    mc_data = mc_img.get_fdata()
    
    # Squeeze if 4D
    if ct_data.ndim == 4: ct_data = np.squeeze(ct_data, -1)
    if spect_data.ndim == 4: spect_data = np.squeeze(spect_data, -1)
    if mc_data.ndim == 4: mc_data = np.squeeze(mc_data, -1)
    
    spacing = ct_img.header.get_zooms()[:3]
    origin = ct_img.affine[:3, 3]
    
    # pytheranostics configuration for Lu177
    config = {
        "Radionuclide": "Lu177",
        "Method": "Voxel-S-Value",
        "OutputFormat": "Dose-Map",
        "PatientID": pat_name,
        "Cycle": 1,
        "InjectedActivity": 7400, # MBq
        "InjectionDate": "20260101",
        "InjectionTime": "110000",
        "ReferenceTimePoint": 0
    }
    
    # We need to provide longitudinal data structures
    nm_study = MockLongStudy(spect_data, spacing, origin)
    ct_study = MockLongStudy(ct_data, spacing, origin)
    
    try:
        # Note: VoxelSDosimetry might require specific library internal structures.
        # If this fails, we will implement a standard DVK convolution manually.
        dose_calc = VoxelSDosimetry(config=config, nm_data=nm_study, ct_data=ct_study)
        dose_calc.compute_tia()
        vsv_dose = dose_calc.results # This should be the 3D dosemap
        
        # Calculate correlation with Monte Carlo
        p = vsv_dose.flatten()
        t = mc_data.flatten()
        if np.std(p) > 1e-6 and np.std(t) > 1e-6:
            c, _ = pearsonr(p, t)
            print(f"  VSV Pearson r: {c:.4f}")
            return c
    except Exception as e:
        print(f"  VoxelSDosimetry failed: {e}")
        # Manually implement standard Lu-177 VSV if library call is too complex
        return None

if __name__ == "__main__":
    data_dir = "/home/user/MedImages.jl/elsarticle/dosimetry/data/"
    val_patients = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])[10:] # Last 2-3 for quick check
    for pat in val_patients:
        generate_vsv_baseline(data_dir, pat)
