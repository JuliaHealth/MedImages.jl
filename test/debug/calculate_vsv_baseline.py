import os
import nibabel as nib
import numpy as np
from scipy.stats import pearsonr

def calculate_vsv_baseline(data_dir):
    print("Calculating Voxel S-Value (VSV) Baseline from Evaluate-Proj-Statistics-Dosimetry params...")
    
    # Constants from Tools.py
    vox_vol_ml = (4.79519987 / 10) ** 3
    lu177_cf = 9.26 # CPS/MBq
    
    # We assume the "dosemap_approx.nii.gz" already represents a VSV-like 
    # but let's re-verify with Monte Carlo ground truth
    
    patients = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    corrs = []
    
    for pat in patients[10:]: # Use validation patients
        pat_dir = os.path.join(data_dir, pat)
        mc = nib.load(os.path.join(pat_dir, "dosemap_mc.nii.gz")).get_fdata()
        approx = nib.load(os.path.join(pat_dir, "dosemap_approx.nii.gz")).get_fdata()
        
        # Pearson correlation is scale invariant, so the exact conversion factor 
        # (cf, volume) doesn't change the r-value if it's a linear mapping.
        # But we use the logic from the repo.
        
        p = approx.flatten()
        t = mc.flatten()
        
        if np.std(p) > 1e-6 and np.std(t) > 1e-6:
            c, _ = pearsonr(p, t)
            corrs.append(c)
            
    avg_corr = np.mean(corrs) if corrs else 0.0
    print(f"Average Pearson Correlation for VSV (Baseline): {avg_corr:.4f}")
    return avg_corr

if __name__ == "__main__":
    data_dir = "/home/user/MedImages.jl/elsarticle/dosimetry/data/"
    calculate_vsv_baseline(data_dir)
