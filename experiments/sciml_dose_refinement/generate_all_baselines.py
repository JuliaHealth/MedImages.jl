import os
import nibabel as nib
import numpy as np
import torch
import joblib
from scipy.ndimage import uniform_filter, distance_transform_edt
import sys

# Add dosimetry dir to sys.path to find baseline_models
sys.path.append(os.path.abspath("elsarticle/dosimetry"))
from baseline_models import Spect0Net, DblurDoseNet

def standardize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

def extract_voxel_features(ct, spect):
    ct_std, sp_std = standardize(ct), standardize(spect)
    ct_mean, sp_mean = uniform_filter(ct_std, size=3), uniform_filter(sp_std, size=3)
    peak_mask = (spect > np.percentile(spect, 95))
    dist = distance_transform_edt(~peak_mask) if np.any(peak_mask) else np.ones_like(spect)*100.0
    return np.stack([sp_std, ct_std, sp_mean, ct_mean, standardize(dist)], axis=-1).reshape(-1, 5)

def generate_missing_binaries():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.abspath("elsarticle/dosimetry")
    pat_dir = os.path.join(base_dir, "data/FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat47")
    out_dir = os.path.join(base_dir, "vis_results_64/FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat47")
    
    # Load NIfTIs
    ct_i = nib.load(os.path.join(pat_dir, "ct.nii.gz")).get_fdata()
    sp_i = nib.load(os.path.join(pat_dir, "spect.nii.gz")).get_fdata()
    ap_i = nib.load(os.path.join(pat_dir, "dosemap_approx.nii.gz")).get_fdata()
    mc_i = nib.load(os.path.join(pat_dir, "dosemap_mc.nii.gz")).get_fdata()
    
    # Squeeze if needed
    if ct_i.ndim == 4: ct_i = np.squeeze(ct_i, -1)
    if sp_i.ndim == 4: sp_i = np.squeeze(sp_i, -1)
    if ap_i.ndim == 4: ap_i = np.squeeze(ap_i, -1)
    if mc_i.ndim == 4: mc_i = np.squeeze(mc_i, -1)

    idx = np.unravel_index(np.argmax(mc_i), mc_i.shape)
    cx, cy, cz = idx; p_s = 64; hx = p_s // 2
    xr = slice(max(0, cx-hx), min(mc_i.shape[0], cx+hx))
    yr = slice(max(0, cy-hx), min(mc_i.shape[1], cy+hx))
    zr = slice(max(0, cz-hx), min(mc_i.shape[2], cz+hx))
    
    ct_p, sp_p, ap_p = ct_i[xr,yr,zr], sp_i[xr,yr,zr], ap_i[xr,yr,zr]
    
    # 1. PBPK-ML
    pbpk = joblib.load(os.path.join(base_dir, "model_baseline_PBPK_ML.joblib"))
    feats = extract_voxel_features(ct_p, sp_p)
    pred_pbpk = pbpk.predict(feats).reshape(p_s, p_s, p_s)
    pred_pbpk.astype(np.float32).tofile(os.path.join(out_dir, "pbpk.bin"))

    # Torch Models
    def run_torch(model_class, path, needs_approx=False):
        m = model_class().to(device)
        m.load_state_dict(torch.load(os.path.join(base_dir, path), map_location=device))
        m.eval()
        with torch.no_grad():
            s = torch.from_numpy(standardize(sp_p)).float().unsqueeze(0).unsqueeze(0).to(device)
            d = torch.from_numpy(standardize(ct_p)).float().unsqueeze(0).unsqueeze(0).to(device)
            if needs_approx:
                a = torch.from_numpy(standardize(ap_p)).float().unsqueeze(0).unsqueeze(0).to(device)
                res = m(s, d, a)
            else:
                res = m(s, d)
        return res.cpu().numpy().squeeze()

    # 2. Spect0
    pred_spect0 = run_torch(Spect0Net, "model_baseline_Spect0.pth", True)
    pred_spect0.astype(np.float32).tofile(os.path.join(out_dir, "spect0.bin"))
    
    # 3. DblurDoseNet
    pred_dblur = run_torch(DblurDoseNet, "model_baseline_DblurDoseNet.pth")
    pred_dblur.astype(np.float32).tofile(os.path.join(out_dir, "dblur.bin"))
    
    # 4. SemiDose
    pred_semi = run_torch(DblurDoseNet, "model_baseline_SemiDose.pth")
    pred_semi.astype(np.float32).tofile(os.path.join(out_dir, "semi.bin"))
    
    print("Missing binaries generated successfully.")

if __name__ == "__main__":
    generate_missing_binaries()
