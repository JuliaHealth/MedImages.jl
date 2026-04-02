import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import rotate

def generate_enhanced_assets():
    vis_dir = "elsarticle/dosimetry/vis_results_64"
    pat_name = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat47"
    pat_dir = os.path.join(vis_dir, pat_name)
    out_dir = "elsarticle/figures_new/clinical_assets"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load 64x64x64 Patches
    dims = (64, 64, 64)
    def load_bin(f):
        return np.fromfile(os.path.join(pat_dir, f), dtype=np.float32).reshape(dims, order='F')

    ct_patch = load_bin("ct.bin")
    ude_patch = load_bin("ude.bin")
    orig_patch = load_bin("orig.bin")
    approx_patch = load_bin("approx.bin")
    cnn_patch = load_bin("cnn.bin")
    
    # --- 1. mip_wholebody.png (Real 512x512x75 CT for whole body feel) ---
    print("Generating mip_wholebody.png from volume-0.nii.gz...")
    img = nib.load('test_data/volume-0.nii.gz')
    full_ct = img.get_fdata()
    # MIP along Y (coronal-like view if standard axial stack)
    # Shape is (512, 512, 75). Let's MIP along axis 1.
    mip_ct = np.max(full_ct, axis=1) 
    plt.figure(figsize=(5, 8))
    # Rotate for portrait MIP view
    plt.imshow(np.rot90(mip_ct), cmap='gray', interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "mip_wholebody.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # --- 2. resampling_ct.png (Using full CT for better resolution) ---
    print("Generating resampling_ct.png...")
    # Take a middle axial slice from full CT
    ct_slice_full = full_ct[:, :, 37]
    ct_rot = rotate(ct_slice_full, 45, reshape=False, order=3)
    plt.figure(figsize=(6, 6))
    plt.imshow(ct_rot, cmap='gray', vmin=-160, vmax=240, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "resampling_ct.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # --- 3. dose_overlay_ct.png (Using the 64x64 patch for authenticity) ---
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

    # --- 4. Challenge 4 Slice Assets (The 3D Metadata Stack - Wide Landscape) ---
    print("Generating Challenge 4 slice assets...")
    def save_wide_slice(data, name, cmap, vmin=None, vmax=None, rot=0):
        plt.figure(figsize=(8, 3))
        s = np.rot90(data[:, :, 32])
        if rot != 0:
            s = rotate(s, rot, reshape=False, order=3)
        # Crop wide
        h, w = s.shape
        wide_s = s[int(h*0.3):int(h*0.7), :]
        plt.imshow(wide_s, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, name), dpi=200, bbox_inches='tight', transparent=True)
        plt.close()

    # Standard slices
    save_wide_slice(ct_patch, "ct_slice.png", "gray", -160, 240)
    save_wide_slice(ude_patch, "dosemap_slice.png", "hot")
    save_wide_slice(orig_patch, "spect_ac_slice.png", "magma")
    save_wide_slice(orig_patch * (1.0 + 0.2 * np.random.randn(*dims)), "spect_nac_slice.png", "magma")

    # Rotated versions (45 degrees)
    save_wide_slice(ct_patch, "ct_slice_rot.png", "gray", -160, 240, rot=45)
    save_wide_slice(ude_patch, "dosemap_slice_rot.png", "hot", rot=45)
    save_wide_slice(orig_patch, "spect_ac_slice_rot.png", "magma", rot=45)
    save_wide_slice(orig_patch * (1.0 + 0.2 * np.random.randn(*dims)), "spect_nac_slice_rot.png", "magma", rot=45)

    # --- 5. Quantitative Lanes (Using Patches) ---
    print("Generating dosimetry comparison lanes...")
    for data, name in [(cnn_patch, "dl_artifacts.png"), 
                       (approx_patch, "vsv_homo.png"), 
                       (ude_patch, "ude_highfi.png")]:
        plt.figure(figsize=(5, 6))
        plt.imshow(np.rot90(data[:, :, cz]), cmap='inferno', interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, name), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

    print(f"\nAll enhanced clinical assets generated in {out_dir}")

if __name__ == "__main__":
    generate_enhanced_assets()
