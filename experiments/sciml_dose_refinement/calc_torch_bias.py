import torch
import torch.nn as nn
import numpy as np
import os
import nibabel as nib
from baseline_models import Spect0Net, DblurDoseNet

def load_pth(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
    return model

def standardize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

def evaluate_bias():
    device = torch.device("cpu")
    data_dir = "/home/user/MedImages.jl/data/dosimetry_data/"
    
    # Load splits
    val_cases = []
    with open("experiments/sciml_dose_refinement/splits.txt", "r") as f:
        in_val = False
        for line in f:
            if "VALIDATION:" in line: in_val = True; continue
            if "TRAINING:" in line: in_val = False; continue
            if in_val and line.strip(): val_cases.append(line.strip())

    models = {
        "Spect0Net": load_pth(Spect0Net(), "data/checkpoints/Spect0/model_baseline_Spect0.pth"),
        "SemiDose": load_pth(DblurDoseNet(), "data/checkpoints/SemiDose/model_baseline_SemiDose.pth"),
        "DblurDoseNet": load_pth(DblurDoseNet(), "data/checkpoints/DblurDoseNet/model_baseline_DblurDoseNet.pth")
    }
    for m in models.values(): m.eval()

    bias_results = {name: [] for name in models}

    for pat in val_cases:
        pat_dir = os.path.join(data_dir, pat)
        if not os.path.isdir(pat_dir): continue
        
        mc_path = os.path.join(pat_dir, "dosemap_mc.nii.gz")
        ct_path = os.path.join(pat_dir, "ct.nii.gz")
        sp_path = os.path.join(pat_dir, "spect.nii.gz")
        ap_path = os.path.join(pat_dir, "dosemap_approx.nii.gz")
        
        if not all(os.path.exists(p) for p in [mc_path, ct_path, sp_path, ap_path]): continue
        
        mc_ni = nib.load(mc_path); mc_raw = mc_ni.get_fdata().squeeze()
        ct_ni = nib.load(ct_path); ct_raw = ct_ni.get_fdata().squeeze()
        sp_ni = nib.load(sp_path); sp_raw = sp_ni.get_fdata().squeeze()
        ap_ni = nib.load(ap_path); ap_raw = ap_ni.get_fdata().squeeze()
        
        # Center crop 64^3
        sz = mc_raw.shape
        cx, cy, cz = sz[0]//2, sz[1]//2, sz[2]//2
        xr, yr, zr = slice(cx-31, cx+33), slice(cy-31, cy+33), slice(cz-31, cz+33)
        
        mc_p = mc_raw[xr, yr, zr]
        ct_p = ct_raw[xr, yr, zr]
        sp_p = sp_raw[xr, yr, zr]
        ap_p = ap_raw[xr, yr, zr]
        
        mask = mc_p > (0.01 * np.max(mc_p))
        
        # Preprocessing matching lu177_data.py
        mc_max = np.max(mc_raw)
        norm_scale = (mc_max / 10.0 + 1e-6)
        
        t_sp = torch.from_numpy(standardize(sp_p)).float().unsqueeze(0).unsqueeze(0)
        t_ct = torch.from_numpy(standardize(ct_p)).float().unsqueeze(0).unsqueeze(0)
        t_ap = torch.from_numpy(standardize(ap_p)).float().unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            for name, model in models.items():
                if name == "Spect0Net":
                    pred = model(t_sp, t_ct, t_ap)
                else:
                    pred = model(t_sp, t_ct)
                
                pred_np = pred.squeeze().numpy()
                # Rescale to mGy (mc_p is in mGy)
                pred_mgy = pred_np * norm_scale
                bias = np.mean(pred_mgy[mask] - mc_p[mask])
                bias_results[name].append(bias)

    print("\nMean Bias [mGy]:")
    for name, biases in bias_results.items():
        print(f"{name}: {np.mean(biases):.2f}")

if __name__ == "__main__":
    evaluate_bias()
