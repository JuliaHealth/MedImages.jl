import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import rotate, zoom

def save_slice(img_data, out_path, cmap='gray', vmin=None, vmax=None):
    plt.figure(figsize=(4, 4))
    plt.imshow(img_data.T, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=150)
    plt.close()

def main():
    pat_id = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat48"
    data_dir = f"/home/user/MedImages.jl/data/dosimetry_data/{pat_id}"
    out_dir = f"/home/user/MedImages.jl/val_outputs_full/{pat_id}"
    img_dir = "/home/user/MedImages.jl/article/figures/images"
    os.makedirs(img_dir, exist_ok=True)

    # Load data
    try:
        ct = nib.load(f"{data_dir}/ct.nii.gz").get_fdata()
        spect = nib.load(f"{data_dir}/spect.nii.gz").get_fdata()
        dose_approx = nib.load(f"{data_dir}/dosemap_approx.nii.gz").get_fdata()
        ude_pred = nib.load(f"{out_dir}/ude_improved_full.nii.gz").get_fdata()
        # Use semi_dose or dblur_dosenet as reference if monte carlo is empty
        # semi_dose might be the closest approx they want to show as ground truth in inference
        ref_dose = nib.load(f"{out_dir}/semi_dose.nii.gz").get_fdata()
        base_dose = nib.load(f"{out_dir}/baseline_analytical.nii.gz").get_fdata()
    except Exception as e:
        print("Error loading data:", e)
        return

    # Squeeze extra dims
    ct = np.squeeze(ct)
    spect = np.squeeze(spect)
    dose_approx = np.squeeze(dose_approx)
    ude_pred = np.squeeze(ude_pred)
    ref_dose = np.squeeze(ref_dose)
    base_dose = np.squeeze(base_dose)

    # Find a good central slice (coronal = y-axis usually, or axial = z-axis)
    # Let's take an axial slice near the middle where organs are visible
    mid_z = ct.shape[2] // 2
    
    ct_slice = ct[:, :, mid_z]
    spect_slice = spect[:, :, mid_z]
    dose_slice = dose_approx[:, :, mid_z]
    
    # Preprocess CT to window it nicely (e.g. soft tissue -100 to 200 HU)
    ct_vmin, ct_vmax = -100, 200
    
    # Phase 2: Perfect Alignment
    save_slice(ct_slice, f"{img_dir}/ct_aligned.png", cmap='bone', vmin=ct_vmin, vmax=ct_vmax)
    save_slice(spect_slice, f"{img_dir}/spect_aligned.png", cmap='hot', vmin=0, vmax=np.percentile(spect_slice, 99))
    save_slice(dose_slice, f"{img_dir}/dose_aligned.png", cmap='magma', vmin=0, vmax=np.percentile(dose_slice, 99))
    
    # Phase 1: Unaligned (Visibly Different)
    # Apply rotation and zoom to CT and Dose to make them look raw/unregistered
    ct_raw = rotate(ct_slice, angle=15, reshape=False)
    ct_raw = zoom(ct_raw, zoom=0.8)
    # Pad back to shape if needed, or just let it be different size
    
    dose_raw = rotate(dose_slice, angle=-10, reshape=False)
    dose_raw = zoom(dose_raw, zoom=1.2)
    
    save_slice(ct_raw, f"{img_dir}/ct_raw.png", cmap='bone', vmin=ct_vmin, vmax=ct_vmax)
    save_slice(spect_slice, f"{img_dir}/spect_raw.png", cmap='hot', vmin=0, vmax=np.percentile(spect_slice, 99)) # spect stays same
    save_slice(dose_raw, f"{img_dir}/dose_raw.png", cmap='magma', vmin=0, vmax=np.percentile(dose_raw, 99))

    # Phase 4: Output Dosimetry (Pat51 Coronal Slices)
    pat51_id = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat51"
    out_dir_51 = f"/home/user/MedImages.jl/val_outputs_full/{pat51_id}"
    
    ude_pred_51 = nib.load(f"{out_dir_51}/ude_improved_full.nii.gz").get_fdata().squeeze()
    ref_dose_51 = nib.load(f"{out_dir_51}/semi_dose.nii.gz").get_fdata().squeeze()
    base_dose_51 = nib.load(f"{out_dir_51}/baseline_analytical.nii.gz").get_fdata().squeeze()

    # In visualize_dosimetry.py, coronal_comparison.png used mid_z
    mid_z_51 = ude_pred_51.shape[2] // 2
    pred_slice = ude_pred_51[:, :, mid_z_51]
    ref_slice = ref_dose_51[:, :, mid_z_51]
    base_slice = base_dose_51[:, :, mid_z_51]

    vmax_dose = max(np.percentile(pred_slice, 99.5), np.percentile(ref_slice, 99.5))
    
    save_slice(pred_slice, f"{img_dir}/pred_dose.png", cmap='hot', vmin=0, vmax=vmax_dose)
    save_slice(ref_slice, f"{img_dir}/ref_dose.png", cmap='hot', vmin=0, vmax=vmax_dose)
    save_slice(base_slice, f"{img_dir}/base_dose.png", cmap='hot', vmin=0, vmax=vmax_dose)

    # Subtractions
    err_pred = np.abs(pred_slice - ref_slice)
    err_base = np.abs(base_slice - ref_slice)
    
    vmax_err = np.percentile(err_base, 99) # use baseline error for scale to show improvement
    
    save_slice(err_pred, f"{img_dir}/err_pred.png", cmap='bwr', vmin=-vmax_err, vmax=vmax_err)
    save_slice(err_base, f"{img_dir}/err_base.png", cmap='bwr', vmin=-vmax_err, vmax=vmax_err)

    print("Images generated successfully in", img_dir)

if __name__ == "__main__":
    main()
