import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import rotate, zoom, gaussian_filter

def get_median_4_corners(img_2d):
    """Computes the median of the 4 corner pixels of a 2D image."""
    h, w = img_2d.shape
    corners = [img_2d[0, 0], img_2d[0, w-1], img_2d[h-1, 0], img_2d[h-1, w-1]]
    return np.median(corners)

def save_fig_precise(name, out_dir, size_px, transparent=True, facecolor=None):
    """Saves a figure with precise pixel dimensions."""
    dpi = 100
    fig = plt.gcf()
    fig.set_size_inches(size_px[0]/dpi, size_px[1]/dpi)
    plt.savefig(os.path.join(out_dir, name), 
                dpi=dpi, 
                bbox_inches='tight', 
                pad_inches=0, 
                transparent=transparent, 
                facecolor=facecolor)
    plt.close()

def resample_to_isotropic(data, spacing, target_spacing=1.0):
    """Resamples a 3D volume to isotropic spacing."""
    scales = [s / target_spacing for s in spacing]
    return zoom(data, scales, order=3)

def generate_final_clinical_assets_v6():
    vis_dir = "elsarticle/dosimetry/vis_results_64"
    pat_name = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat47"
    pat_dir = os.path.join(vis_dir, pat_name)
    out_dir = "elsarticle/figures_new/clinical_assets"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load and Resample Full CT Volume
    print("Loading and resampling full CT volume...")
    full_ct_img = nib.load('test_data/volume-0.nii.gz')
    full_ct_raw = full_ct_img.get_fdata()
    spacing_full = full_ct_img.header.get_zooms()[:3]
    full_ct = resample_to_isotropic(full_ct_raw, spacing_full)
    
    # 2. Load and Resample 64x64x64 Patches
    # Assume SPECT spacing ~4.8mm
    spacing_patch = (4.795, 4.795, 4.795)
    dims = (64, 64, 64)
    def load_bin_resampled(f):
        data = np.fromfile(os.path.join(pat_dir, f), dtype=np.float32).reshape(dims, order='F')
        return resample_to_isotropic(data, spacing_patch)

    ct_patch = load_bin_resampled("ct.bin")
    ude_patch = load_bin_resampled("ude.bin")
    orig_patch = load_bin_resampled("orig.bin")
    approx_patch = load_bin_resampled("approx.bin")
    cnn_patch = load_bin_resampled("cnn.bin")
    
    # --- 1. mip_wholebody.png (Enhanced Body Outline) ---
    print("Generating mip_wholebody.png...")
    mip_ct = np.max(full_ct, axis=1) # Coronal MIP
    vmin, vmax = -700, 1000 # Lighter window
    mip_norm = np.clip((mip_ct - vmin) / (vmax - vmin), 0, 1)
    plt.figure()
    plt.imshow(np.rot90(mip_norm), cmap='Greys', interpolation='bicubic')
    plt.axis('off')
    save_fig_precise("mip_wholebody.png", out_dir, (100, 120), transparent=False, facecolor='white')

    # --- 2. resampling_ct.png (Workflow from transformation_consistency: Sagittal Full) ---
    print("Generating resampling_ct.png (Sagittal)...")
    # Central sagittal slice
    ct_slice_sag = full_ct[full_ct.shape[0]//2, :, :]
    extrap = get_median_4_corners(ct_slice_sag)
    # 45-degree rotation with median extrapolation
    ct_rot = rotate(ct_slice_sag, 45, reshape=False, order=3, mode='constant', cval=extrap)
    plt.figure()
    plt.imshow(np.rot90(ct_rot), cmap='gray', vmin=-160, vmax=240, interpolation='bilinear')
    plt.axis('off')
    save_fig_precise("resampling_ct.png", out_dir, (100, 100))

    # --- 3. dose_overlay_ct.png (Sagittal Patch Overlay) ---
    print("Generating dose_overlay_ct.png...")
    plt.figure()
    mid_x_p = ct_patch.shape[0] // 2
    plt.imshow(np.rot90(ct_patch[mid_x_p, :, :]), cmap='gray', vmin=-160, vmax=240)
    dose_mask = np.rot90(ude_patch[mid_x_p, :, :])
    dose_mask = np.ma.masked_where(dose_mask < 0.05 * np.max(dose_mask), dose_mask)
    plt.imshow(dose_mask, cmap='hot', alpha=0.6, interpolation='bilinear')
    plt.axis('off')
    save_fig_precise("dose_overlay_ct.png", out_dir, (80, 100))

    # --- 4. Challenge 4 Slice Assets (Full Sagittal Stacks) ---
    print("Generating Challenge 4 stacks (Sagittal)...")
    def save_sag_stack(img_2d, name, cmap, vmin=None, vmax=None, rot=0):
        extrap = get_median_4_corners(img_2d)
        if rot != 0:
            img_2d = rotate(img_2d, rot, reshape=False, order=3, mode='constant', cval=extrap)
        img_final = np.rot90(img_2d)
        plt.figure()
        plt.imshow(img_final, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
        plt.axis('off')
        # Use a taller aspect ratio for sagittal slices in the stack
        save_fig_precise(name, out_dir, (120, 150))

    # Master CT slices
    master_sag_ct = full_ct[full_ct.shape[0]//2, :, :]
    save_sag_stack(master_sag_ct, "ct_slice.png", "gray", -160, 240)
    save_sag_stack(master_sag_ct, "ct_slice_rot.png", "gray", -160, 240, rot=45)
    
    # Rescale patches to match master CT sagittal dimensions
    def prep_sag_patch(p):
        s = p[p.shape[0]//2, :, :]
        return zoom(s, (full_ct.shape[1]/p.shape[1], full_ct.shape[2]/p.shape[2]), order=3)

    res_ude = prep_sag_patch(ude_patch)
    save_sag_stack(res_ude, "dosemap_slice.png", "hot")
    save_sag_stack(res_ude, "dosemap_slice_rot.png", "hot", rot=45)
    
    res_orig = prep_sag_patch(orig_patch)
    save_sag_stack(res_orig, "spect_ac_slice.png", "magma")
    save_sag_stack(res_orig, "spect_ac_slice_rot.png", "magma", rot=45)
    
    res_nac = prep_sag_patch(orig_patch * (1.0 + 0.15 * np.random.randn(*orig_patch.shape)))
    save_sag_stack(res_nac, "spect_nac_slice.png", "magma")
    save_sag_stack(res_nac, "spect_nac_slice_rot.png", "magma", rot=45)

    # --- 5. Quantitative Lanes (Sagittal Patches) ---
    print("Generating dosimetry comparison lanes...")
    for p, name in [(cnn_patch, "dl_artifacts.png"), 
                   (approx_patch, "vsv_homo.png"), 
                   (ude_patch, "ude_highfi.png")]:
        plt.figure()
        plt.imshow(np.rot90(p[p.shape[0]//2, :, :]), cmap='inferno', interpolation='bilinear')
        plt.axis('off')
        save_fig_precise(name, out_dir, (80, 100))

    print(f"\nSUCCESS: All clinical assets generated in {out_dir}")

if __name__ == "__main__":
    generate_final_clinical_assets_v6()
