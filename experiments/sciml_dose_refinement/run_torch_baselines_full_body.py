import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import glob
from tqdm import tqdm
from baseline_models import Spect0Net, DblurDoseNet

def apply_sliding_window(spect, density, approx, model, model_type="dblur"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    nx, ny, nz = spect.shape
    stride = 48
    p_s = 64
    
    x_centers = sorted(list(set(list(range(0, nx - p_s + 1, stride)) + [nx - p_s])))
    y_centers = sorted(list(set(list(range(0, ny - p_s + 1, stride)) + [ny - p_s])))
    z_centers = sorted(list(set(list(range(0, nz - p_s + 1, stride)) + [nz - p_s])))
    
    out = np.zeros_like(spect, dtype=np.float32)
    counts = np.zeros_like(spect, dtype=np.float32)
    
    # Cosine window
    w = 1.0 - np.cos(np.linspace(0, 2*np.pi, p_s))
    w = w / (w.max() + 1e-6)
    win3d = w[:, None, None] * w[None, :, None] * w[None, None, :]
    
    with torch.no_grad():
        for i in tqdm(x_centers, desc=f"Reconstructing {model_type}..."):
            for j in y_centers:
                for k in z_centers:
                    s_p = spect[i:i+p_s, j:j+p_s, k:k+p_s]
                    if s_p.sum() < 1e-2: continue
                    
                    d_p = density[i:i+p_s, j:j+p_s, k:k+p_s]
                    a_p = approx[i:i+p_s, j:j+p_s, k:k+p_s]
                    
                    s_t = torch.from_numpy(s_p).float().to(device).unsqueeze(0).unsqueeze(0)
                    d_t = torch.from_numpy(d_p).float().to(device).unsqueeze(0).unsqueeze(0)
                    a_t = torch.from_numpy(a_p).float().to(device).unsqueeze(0).unsqueeze(0)
                    
                    if model_type == "spect0":
                        pred = model(s_t, d_t, a_t)
                    else:
                        pred = model(s_t, d_t)
                        
                    res = pred.squeeze().cpu().numpy()
                    out[i:i+p_s, j:j+p_s, k:k+p_s] += res * win3d
                    counts[i:i+p_s, j:j+p_s, k:k+p_s] += win3d
                    
    return out / (counts + 1e-6)

def run():
    pat_dirs = glob.glob("val_outputs/FDM*")
    
    for out_dir in pat_dirs:
        pat_name = os.path.basename(out_dir)
        pat_data_dir = f"data/dosimetry_data/{pat_name}"
        
        if not os.path.exists(os.path.join(pat_data_dir, "spect.nii.gz")):
            print(f"Skipping {pat_name}, source data missing.")
            continue
            
        print(f"--- Processing {pat_name} ---")
        
        spect = nib.load(os.path.join(pat_data_dir, "spect.nii.gz")).get_fdata()
        ct = nib.load(os.path.join(pat_data_dir, "ct.nii.gz")).get_fdata()
        approx = nib.load(os.path.join(pat_data_dir, "dosemap_approx.nii.gz")).get_fdata()
        
        if spect.ndim == 4: spect = spect[..., 0]
        if ct.ndim == 4: ct = ct[..., 0]
        if approx.ndim == 4: approx = approx[..., 0]
        
        # Normalize inputs for Torch baselines
        s_std = (spect - spect.mean()) / (spect.std() + 1e-6)
        # HU to density
        density = np.where(ct <= 0, np.maximum(0.01, 1.0 + 0.001 * ct), 1.0 + 0.0007 * ct)
        d_std = (density - density.mean()) / (density.std() + 1e-6)
        a_std = (approx - approx.mean()) / (approx.std() + 1e-6)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. DblurDoseNet
        m_dblur = DblurDoseNet().to(device)
        m_dblur.load_state_dict(torch.load("data/checkpoints/DblurDoseNet/model_baseline_DblurDoseNet.pth", map_location=device))
        res_dblur = apply_sliding_window(s_std, d_std, a_std, m_dblur, "dblur")
        res_dblur.astype(np.float32).tofile(os.path.join(out_dir, "full_dblur.bin"))

        # 2. SemiDose
        m_semi = DblurDoseNet().to(device)
        m_semi.load_state_dict(torch.load("data/checkpoints/SemiDose/model_baseline_SemiDose.pth", map_location=device))
        res_semi = apply_sliding_window(s_std, d_std, a_std, m_semi, "semi")
        res_semi.astype(np.float32).tofile(os.path.join(out_dir, "full_semi.bin"))

        # 3. Spect0Net
        m_spect0 = Spect0Net().to(device)
        m_spect0.load_state_dict(torch.load("data/checkpoints/Spect0/model_baseline_Spect0.pth", map_location=device))
        res_spect0 = apply_sliding_window(s_std, d_std, a_std, m_spect0, "spect0")
        res_spect0.astype(np.float32).tofile(os.path.join(out_dir, "full_spect0.bin"))

if __name__ == "__main__":
    run()
