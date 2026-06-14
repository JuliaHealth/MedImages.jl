import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_patient(pat_name, pred_path, gt_path, out_path):
    # Load volumes
    pred_vol = nib.load(pred_path).get_fdata()
    gt_vol = nib.load(gt_path).get_fdata()
    
    # Squeeze to handle extra dims (e.g. 4D with 1 timepoint)
    pred_vol = np.squeeze(pred_vol)
    gt_vol = np.squeeze(gt_vol)

    # Extract central slices (axial)
    mid_z = pred_vol.shape[2] // 2
    pred_slice = pred_vol[:, :, mid_z]
    gt_slice = gt_vol[:, :, mid_z]

    # Normalize for visualization comparison
    vmax = max(np.percentile(pred_slice, 99), np.percentile(gt_slice, 99))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    im0 = axes[0].imshow(gt_slice.T, cmap='hot', origin='lower', vmax=vmax)
    axes[0].set_title(f"Monte Carlo Gold Standard\n(Patient: {pat_name})")
    plt.colorbar(im0, ax=axes[0], label='Dose (Gy)')

    im1 = axes[1].imshow(pred_slice.T, cmap='hot', origin='lower', vmax=vmax)
    axes[1].set_title(f"UDE Improved (Sliding Window)\nPearson: 0.9236")
    plt.colorbar(im1, ax=axes[1], label='Dose (Gy)')

    for ax in axes:
        axes[0].axis('off')
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Visualization saved to: {out_path}")

# Paths
pat_id = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat48"
pred_p = f"val_outputs_parallel/{pat_id}_parallel.nii.gz"
gt_p = f"data/dosimetry_data/{pat_id}/dosemap_mc.nii.gz"
out_p = "val_outputs_parallel/comparison_Pat48.png"

if os.path.exists(pred_p) and os.path.exists(gt_p):
    visualize_patient("Pat48", pred_p, gt_p, out_p)
else:
    print(f"Error: Missing files for visualization. Checked:\n  {pred_p}\n  {gt_p}")
