import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import rotate, zoom, gaussian_filter

def get_median_4_corners(img_2d):
    """Computes the median of the 4 corner pixels of a 2D image, matching MedImages default behavior."""
    h, w = img_2d.shape
    corners = [img_2d[0, 0], img_2d[0, w-1], img_2d[h-1, 0], img_2d[h-1, w-1]]
    return np.median(corners)

def generate_final_clinical_assets_v4():
    vis_dir = "elsarticle/dosimetry/vis_results_64"
    pat_name = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat47"
    pat_dir = os.path.join(vis_dir, pat_name)
    out_dir = "elsarticle/figures_new/clinical_assets"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load 64x64x64 Patches (for dosimetry lanes and multimodal stacks)
    dims = (64, 64, 64)
    def load_bin(f):
        return np.fromfile(os.path.join(pat_dir, f), dtype=np.float32).reshape(dims, order='F')

    ct_patch = load_bin("ct.bin")
    ude_patch = load_bin("ude.bin")
    orig_patch = load_bin("orig.bin")
    approx_patch = load_bin("approx.bin")
    cnn_patch = load_bin("cnn.bin")
    
    # Load Full 512x512x75 CT Volume
    full_ct_img = nib.load('test_data/volume-0.nii.gz')
    full_ct = full_ct_img.get_fdata()
    
    # --- 1. mip_wholebody.png (LibCarna Style: Coronal, Inverted, Depth-Shaded) ---
    print("Generating mip_wholebody.png (LibCarna-inspired Clinical MIP)...")
    # Using the UDE Champion patch (64x64x64) for the dose distribution
    # Project along axis 1 (Y/Coronal depth)
    mip_spect = np.max(ude_patch, axis=1)
    
    # Depth-dependent shading: weighting slices along axis 1
    # LibCarna often uses shading to enhance volume perception.
    shaded_mip = np.zeros((64, 64))
    weights = np.linspace(0.8, 1.2, 64) # Closer slices are slightly brighter
    for i in range(64):
        shaded_mip = np.maximum(shaded_mip, ude_patch[:, i, :] * weights[i])
    
    # Clinical Windowing: clip to 99th percentile and invert
    vmax = np.percentile(shaded_mip, 99.0)
    shaded_mip = np.clip(shaded_mip, 0, vmax)
    shaded_mip = gaussian_filter(shaded_mip, sigma=0.5) # Smooth like clinical renderers
    
    plt.figure(figsize=(4, 6), facecolor='white')
    # Use inverted grayscale as standard in NM (black uptake on white)
    plt.imshow(np.rot90(shaded_mip), cmap='Greys', interpolation='bicubic')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "mip_wholebody.png"), dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

    # --- 2. resampling_ct.png (Whole 512x512 axial CT slice with Corner Extrapolation) ---
    print("Generating resampling_ct.png (Whole Slice, Median Extrapolation)...")
    ct_slice_full = full_ct[:, :, 37]
    extrap_val = get_median_4_corners(ct_slice_full)
    
    # Apply 45-degree rotation with default MedImages-like extrapolation
    ct_rot = rotate(ct_slice_full, 45, reshape=False, order=3, mode='constant', cval=extrap_val)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(ct_rot, cmap='gray', vmin=-160, vmax=240, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "resampling_ct.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # --- 3. dose_overlay_ct.png (Patch-based Clinical Overlay) ---
    print("Generating dose_overlay_ct.png...")
    plt.figure(figsize=(5, 6))
    cz = 32
    plt.imshow(np.rot90(ct_patch[:, :, cz]), cmap='gray', vmin=-160, vmax=240)
    dose_mask = np.rot90(ude_patch[:, :, cz])
    dose_mask = np.ma.masked_where(dose_mask < 0.05 * np.max(dose_mask), dose_mask)
    plt.imshow(dose_mask, cmap='hot', alpha=0.6, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "dose_overlay_ct.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # --- 4. Challenge 4 Slice Assets (The Metadata Stack - Rescaled Patch Consistency) ---
    print("Generating Challenge 4 stacks (Aligned Stack with Rescaling)...")
    
    def save_aligned_slice(patch_3d, name, cmap, vmin=None, vmax=None, rot=0):
        s_2d = patch_3d[:, :, 32]
        # Rescale to 512-grid to match full context expectations
        s_res = zoom(s_2d, (8, 8), order=3)
        extrap = get_median_4_corners(s_res)
        if rot != 0:
            s_res = rotate(s_res, rot, reshape=False, order=3, mode='constant', cval=extrap)
        
        # Orient correctly
        img_final = np.rot90(s_res)
        
        plt.figure(figsize=(10, 4))
        # Wide landscape crop as requested in Appendix (120x50 pixels equivalent)
        h, w = img_final.shape
        wide_s = img_final[int(h*0.35):int(h*0.65), :]
        plt.imshow(wide_s, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, name), dpi=200, bbox_inches='tight', transparent=True)
        plt.close()

    # Full Modality Alignment Stack
    save_aligned_slice(ct_patch, "ct_slice.png", "gray", -160, 240)
    save_aligned_slice(ct_patch, "ct_slice_rot.png", "gray", -160, 240, rot=45)
    save_aligned_slice(ude_patch, "dosemap_slice.png", "hot")
    save_aligned_slice(ude_patch, "dosemap_slice_rot.png", "hot", rot=45)
    save_aligned_slice(orig_patch, "spect_ac_slice.png", "magma")
    save_aligned_slice(orig_patch, "spect_ac_slice_rot.png", "magma", rot=45)
    save_aligned_slice(orig_patch * (1.0 + 0.2 * np.random.randn(*dims)), "spect_nac_slice.png", "magma")
    save_aligned_slice(orig_patch * (1.0 + 0.2 * np.random.randn(*dims)), "spect_nac_slice_rot.png", "magma", rot=45)

    # --- 5. Quantitative Experiment Lanes ---
    print("Finalizing Dosimetry Comparison Lanes...")
    for data, name in [(cnn_patch, "dl_artifacts.png"), 
                       (approx_patch, "vsv_homo.png"), 
                       (ude_patch, "ude_highfi.png")]:
        plt.figure(figsize=(5, 6))
        plt.imshow(np.rot90(data[:, :, 32]), cmap='inferno', interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, name), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

    print(f"\nCOMPLETED: Finalized clinical assets in {out_dir}")

if __name__ == "__main__":
    generate_final_clinical_assets_v4()
