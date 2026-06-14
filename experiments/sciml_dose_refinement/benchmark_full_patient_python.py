import os
import sys
import nibabel as nib
import numpy as np
import torch
import time

sys.path.append(os.path.abspath("experiments/sciml_dose_refinement"))
sys.path.append(os.path.abspath("elsarticle/dosimetry"))
from baseline_models import DblurDoseNet

def standardize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

def get_3d_cosine_window(size):
    w = 1.0 - np.cos(np.linspace(0, 2*np.pi, size))
    w = w / np.max(w)
    win2d = np.outer(w, w)
    win3d = win2d[..., None] * w[None, None, :]
    return win3d.astype(np.float32)

def sliding_window_inference(model, spect, ct, device, patch_size=64, stride=32):
    nx, ny, nz = spect.shape
    final_dose = np.zeros((nx, ny, nz), dtype=np.float32)
    counts = np.zeros((nx, ny, nz), dtype=np.float32)
    window = get_3d_cosine_window(patch_size)
    
    x_centers = list(range(0, nx - patch_size + 1, stride)); x_centers.append(nx - patch_size)
    y_centers = list(range(0, ny - patch_size + 1, stride)); y_centers.append(ny - patch_size)
    z_centers = list(range(0, nz - patch_size + 1, stride)); z_centers.append(nz - patch_size)
    
    x_centers = sorted(list(set(x_centers)))
    y_centers = sorted(list(set(y_centers)))
    z_centers = sorted(list(set(z_centers)))
    
    for i in x_centers:
        for j in y_centers:
            for k in z_centers:
                sp_patch = spect[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                ct_patch = ct[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                
                if np.sum(sp_patch) < 1e-2:
                    continue
                
                s = torch.from_numpy(standardize(sp_patch)).float().unsqueeze(0).unsqueeze(0).to(device)
                d = torch.from_numpy(standardize(ct_patch)).float().unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred = model(s, d).cpu().numpy().squeeze()
                
                final_dose[i:i+patch_size, j:j+patch_size, k:k+patch_size] += pred * window
                counts[i:i+patch_size, j:j+patch_size, k:k+patch_size] += window
                
    return final_dose / (counts + 1e-6)

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    m_dblur = DblurDoseNet().to(device)
    m_dblur.eval()

    pat_name = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat48"
    dataset_dir = "data/dosimetry_data"
    pat_data_dir = os.path.join(dataset_dir, pat_name)
    
    ct_i = nib.load(os.path.join(pat_data_dir, "ct.nii.gz")).get_fdata()
    sp_i = nib.load(os.path.join(pat_data_dir, "spect.nii.gz")).get_fdata()
    
    ct_i = np.squeeze(ct_i)
    sp_i = np.squeeze(sp_i)
    
    print("Warmup...")
    dummy_sp = np.zeros((64, 64, 64), dtype=np.float32); dummy_sp[32,32,32] = 1.0
    dummy_ct = np.zeros((64, 64, 64), dtype=np.float32)
    s = torch.from_numpy(dummy_sp).float().unsqueeze(0).unsqueeze(0).to(device)
    d = torch.from_numpy(dummy_ct).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = m_dblur(s, d)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    print("Benchmarking full patient sliding window (PyTorch DL)...")
    
    t0 = time.time()
    sliding_window_inference(m_dblur, sp_i, ct_i, device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t1 = time.time()
    
    print(f"Full Patient Inference Time (PyTorch DL): {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    run()