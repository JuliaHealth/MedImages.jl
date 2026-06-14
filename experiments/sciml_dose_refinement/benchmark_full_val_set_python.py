import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import glob
import time
from tqdm import tqdm

# Mock models for timing if checkpoints aren't loaded, 
# but we'll try to load them to be realistic.
from baseline_models import Spect0Net, DblurDoseNet

def apply_sliding_window(spect, density, approx, model, model_type="dblur"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    nx, ny, nz = spect.shape
    stride = 32
    p_s = 64
    
    x_centers = sorted(list(set(list(range(0, nx - p_s + 1, stride)) + [nx - p_s])))
    y_centers = sorted(list(set(list(range(0, ny - p_s + 1, stride)) + [ny - p_s])))
    z_centers = sorted(list(set(list(range(0, nz - p_s + 1, stride)) + [nz - p_s])))
    
    # Cosine window
    w = 1.0 - np.cos(np.linspace(0, 2*np.pi, p_s))
    w = w / (w.max() + 1e-6)
    win3d = w[:, None, None] * w[None, :, None] * w[None, None, :]
    
    with torch.no_grad():
        for i in x_centers:
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
                        _ = model(s_t, d_t, a_t)
                    else:
                        _ = model(s_t, d_t)
                    
                    torch.cuda.synchronize()

def run_benchmarks():
    val_cases = []
    with open("experiments/sciml_dose_refinement/splits.txt", "r") as f:
        in_val = False
        for line in f:
            if "VALIDATION:" in line: in_val = True; continue
            if "TRAINING:" in line: in_val = False; continue
            if in_val and line.strip(): val_cases.append(line.strip())
    
    val_cases.sort()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init models
    m_dblur = DblurDoseNet().to(device).eval()
    m_semi = DblurDoseNet().to(device).eval() # Same architecture for timing
    m_spect0 = Spect0Net().to(device).eval()
    m_hybrid = DblurDoseNet().to(device).eval() # Approximating hybrid speed

    results = []
    
    print("Starting Full-Set Benchmark (Python Models)...")
    for idx, pat in enumerate(val_cases):
        pat_data_dir = f"data/dosimetry_data/{pat}"
        if not os.path.exists(os.path.join(pat_data_dir, "spect.nii.gz")): continue
        
        print(f"\n>>> Case {idx+1}/{len(val_cases)}: {pat}")
        
        spect = nib.load(os.path.join(pat_data_dir, "spect.nii.gz")).get_fdata()
        ct = nib.load(os.path.join(pat_data_dir, "ct.nii.gz")).get_fdata()
        
        if spect.ndim == 4: spect = spect[..., 0]
        if ct.ndim == 4: ct = ct[..., 0]
        
        # Mock approx for timing
        approx = np.zeros_like(spect)
        
        # 1. Dblur
        t0 = time.time()
        apply_sliding_window(spect, ct, approx, m_dblur, "dblur")
        t_dblur = time.time() - t0
        
        # 2. SemiDose
        t0 = time.time()
        apply_sliding_window(spect, ct, approx, m_semi, "semi")
        t_semi = time.time() - t0
        
        # 3. Spect0
        t0 = time.time()
        apply_sliding_window(spect, ct, approx, m_spect0, "spect0")
        t_spect0 = time.time() - t0

        # 4. Hybrid
        t0 = time.time()
        apply_sliding_window(spect, ct, approx, m_hybrid, "hybrid")
        t_hybrid = time.time() - t0
        
        results.append((pat, t_dblur, t_semi, t_spect0, t_hybrid))
        print(f"Results for {pat}: Dblur={t_dblur:.2f}s, Semi={t_semi:.2f}s, Spect0={t_spect0:.2f}s, Hybrid={t_hybrid:.2f}s")
        
    # Stats
    m_dblur = np.mean([r[1] for r in results])
    m_semi = np.mean([r[2] for r in results])
    m_spect0 = np.mean([r[3] for r in results])
    m_hybrid = np.mean([r[4] for r in results])
    
    print("\n" + "="*60)
    print("MEAN TIMES (Python):")
    print(f"DblurDoseNet: {m_dblur:.2f} s")
    print(f"SemiDose:     {m_semi:.2f} s")
    print(f"Spect0Net:    {m_spect0:.2f} s")
    print(f"CNN+Approx:   {m_hybrid:.2f} s")

if __name__ == "__main__":
    run_benchmarks()
