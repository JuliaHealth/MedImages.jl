import os
import sys
import nibabel as nib
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath("experiments/sciml_dose_refinement"))
sys.path.append(os.path.abspath("elsarticle/dosimetry"))
try:
    from baseline_models import Spect0Net, DblurDoseNet
except ImportError:
    print("Could not import baseline_models. Make sure it is in path.")
    sys.exit(1)

def standardize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

def get_3d_cosine_window(size):
    w = 1.0 - np.cos(np.linspace(0, 2*np.pi, size))
    w = w / np.max(w)
    win2d = np.outer(w, w)
    win3d = win2d[..., None] * w[None, None, :]
    return win3d.astype(np.float32)

def sliding_window_inference(model, spect, ct, approx, needs_approx, device, patch_size=64, stride=32):
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
                
                # Check sum
                if np.sum(sp_patch) < 1e-2:
                    continue
                
                s = torch.from_numpy(standardize(sp_patch)).float().unsqueeze(0).unsqueeze(0).to(device)
                d = torch.from_numpy(standardize(ct_patch)).float().unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    if needs_approx:
                        ap_patch = approx[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                        a = torch.from_numpy(standardize(ap_patch)).float().unsqueeze(0).unsqueeze(0).to(device)
                        pred = model(s, d, a).cpu().numpy().squeeze()
                    else:
                        pred = model(s, d).cpu().numpy().squeeze()
                
                final_dose[i:i+patch_size, j:j+patch_size, k:k+patch_size] += pred * window
                counts[i:i+patch_size, j:j+patch_size, k:k+patch_size] += window
                
    return final_dose / (counts + 1e-6)

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    m_dblur = DblurDoseNet().to(device)
    m_dblur.load_state_dict(torch.load("data/checkpoints/DblurDoseNet/model_baseline_DblurDoseNet.pth", map_location=device))
    m_dblur.eval()

    m_semi = DblurDoseNet().to(device) # SemiDose uses same architecture
    m_semi.load_state_dict(torch.load("data/checkpoints/SemiDose/model_baseline_SemiDose.pth", map_location=device))
    m_semi.eval()

    m_spect0 = Spect0Net().to(device)
    m_spect0.load_state_dict(torch.load("data/checkpoints/Spect0/model_baseline_Spect0.pth", map_location=device))
    m_spect0.eval()

    val_out_root = "val_outputs_full"
    dataset_dir = "data/dosimetry_data"
    
    pats = [d for d in os.listdir(val_out_root) if os.path.isdir(os.path.join(val_out_root, d)) and d.startswith("FDM")]
    
    for pat in tqdm(pats):
        pat_out_dir = os.path.join(val_out_root, pat)
        pat_data_dir = os.path.join(dataset_dir, pat)
        
        ct_i = nib.load(os.path.join(pat_data_dir, "ct.nii.gz")).get_fdata()
        sp_i = nib.load(os.path.join(pat_data_dir, "spect.nii.gz")).get_fdata()
        mc_i = nib.load(os.path.join(pat_data_dir, "dosemap_mc.nii.gz")).get_fdata()
        ap_i = nib.load(os.path.join(pat_data_dir, "dosemap_approx.nii.gz")).get_fdata()
        
        ct_i = np.squeeze(ct_i)
        sp_i = np.squeeze(sp_i)
        mc_i = np.squeeze(mc_i)
        ap_i = np.squeeze(ap_i)
        
        pred_dblur = np.squeeze(nib.load(os.path.join(pat_out_dir, "dblur_dosenet.nii.gz")).get_fdata())
        pred_semi = np.squeeze(nib.load(os.path.join(pat_out_dir, "semi_dose.nii.gz")).get_fdata())
        pred_spect0 = np.squeeze(nib.load(os.path.join(pat_out_dir, "spect0.nii.gz")).get_fdata())
        
        # Determine informative coronal slice (y-axis)
        # We want the coronal slice with the highest variance in the ground truth
        coronal_variances = [np.var(mc_i[:, y, :]) for y in range(mc_i.shape[1])]
        best_y = np.argmax(coronal_variances)
        
        ude_i = nib.load(os.path.join(pat_out_dir, "ude_improved_full.nii.gz")).get_fdata()
        base_i = nib.load(os.path.join(pat_out_dir, "baseline_analytical.nii.gz")).get_fdata()
        
        ude_i = np.squeeze(ude_i)
        base_i = np.squeeze(base_i)
        
        s_ct = ct_i[:, best_y, :]
        
        # MIP extraction for dose (along coronal axis)
        mip_mc = np.max(mc_i, axis=1)
        mip_ude = np.max(ude_i, axis=1)
        mip_base = np.max(base_i, axis=1)
        mip_dblur = np.max(pred_dblur, axis=1)
        mip_semi = np.max(pred_semi, axis=1)
        mip_spect0 = np.max(pred_spect0, axis=1)
        
        vmax_dose = np.percentile(mip_mc, 99.5)
        if vmax_dose == 0: vmax_dose = 1.0
        
        mip_base_norm = mip_base * (vmax_dose / (np.max(mip_base)+1e-6))
        
        mip_sub = mip_mc - mip_ude
        mip_sub_base = mip_mc - mip_base_norm
        
        vmax_sub = max(np.percentile(np.abs(mip_sub), 99.5), np.percentile(np.abs(mip_sub_base), 99.5))
        if vmax_sub == 0: vmax_sub = 1.0
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        
        axes[0, 0].imshow(np.rot90(s_ct), cmap='gray')
        axes[0, 0].set_title(f"CT Context\nCoronal Slice Y={best_y}")
        axes[0, 0].axis('off')
        
        im0 = axes[0, 1].imshow(np.rot90(mip_mc), cmap='hot', vmin=0, vmax=vmax_dose)
        axes[0, 1].set_title("Monte Carlo (GT) MIP")
        axes[0, 1].axis('off')
        
        im1 = axes[0, 2].imshow(np.rot90(mip_base_norm), cmap='hot', vmin=0, vmax=vmax_dose)
        axes[0, 2].set_title("Analytical Baseline MIP")
        axes[0, 2].axis('off')
        
        mip_spect0_norm = mip_spect0 * (vmax_dose / (np.max(mip_spect0)+1e-6))
        im2 = axes[1, 0].imshow(np.rot90(mip_spect0_norm), cmap='hot', vmin=0, vmax=vmax_dose)
        axes[1, 0].set_title("Spect0 MIP")
        axes[1, 0].axis('off')

        mip_dblur_norm = mip_dblur * (vmax_dose / (np.max(mip_dblur)+1e-6))
        im3 = axes[1, 1].imshow(np.rot90(mip_dblur_norm), cmap='hot', vmin=0, vmax=vmax_dose)
        axes[1, 1].set_title("DblurDoseNet MIP")
        axes[1, 1].axis('off')
        
        mip_semi_norm = mip_semi * (vmax_dose / (np.max(mip_semi)+1e-6))
        im4 = axes[1, 2].imshow(np.rot90(mip_semi_norm), cmap='hot', vmin=0, vmax=vmax_dose)
        axes[1, 2].set_title("SemiDose MIP")
        axes[1, 2].axis('off')

        im5 = axes[2, 0].imshow(np.rot90(mip_ude), cmap='hot', vmin=0, vmax=vmax_dose)
        axes[2, 0].set_title("UDE Improved (Ours) MIP")
        axes[2, 0].axis('off')
        
        im_sub = axes[2, 1].imshow(np.rot90(mip_sub), cmap='coolwarm', vmin=-vmax_sub, vmax=vmax_sub)
        axes[2, 1].set_title("Subtraction (MC - UDE) MIP")
        axes[2, 1].axis('off')
        
        im_sub_base = axes[2, 2].imshow(np.rot90(mip_sub_base), cmap='coolwarm', vmin=-vmax_sub, vmax=vmax_sub)
        axes[2, 2].set_title("Subtraction (MC - Baseline) MIP")
        axes[2, 2].axis('off')
        
        plt.suptitle(f"Patient: {pat}", fontsize=20, y=0.92)
        plt.savefig(os.path.join(pat_out_dir, "coronal_comparison.png"), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

if __name__ == "__main__":
    run()