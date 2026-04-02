import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import rotate, zoom

def generate_final_clinical_assets():
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
    
    # Load 512x512x75 CT for "Whole Slices"
    full_ct_img = nib.load('test_data/volume-0.nii.gz')
    full_ct = full_ct_img.get_fdata()
    
    # --- 1. mip_wholebody.png (Clinical Standard SPECT MIP) ---
    # According to clinical standards: Coronal view, inverted Greys, 99.5% windowing
    # We MIP the UDE Champion patch
    print("Generating mip_wholebody.png (Clinical SPECT MIP)...")
    # Axis 1 is depth (Y), so projecting along Y gives Coronal (X-Z)
    mip_spect = np.max(ude_patch, axis=1)
    # Clinical Windowing: clip to 99.5th percentile
    vmax = np.percentile(mip_spect, 99.5)
    mip_spect = np.clip(mip_spect, 0, vmax)
    # Visualization: Inverted Greys (black lesions on white)
    plt.figure(figsize=(4, 6), facecolor='white')
    # Rotate for portrait view (assuming Z is axis 2)
    plt.imshow(np.rot90(mip_spect), cmap='Greys', interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "mip_wholebody.png"), dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

    # --- 2. resampling_ct.png (Whole 512x512 axial CT slice) ---
    print("Generating resampling_ct.png (Whole Slice)...")
    ct_slice_full = full_ct[:, :, 37]
    ct_rot = rotate(ct_slice_full, 45, reshape=False, order=3)
    plt.figure(figsize=(6, 6))
    plt.imshow(ct_rot, cmap='gray', vmin=-160, vmax=240, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "resampling_ct.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # --- 3. dose_overlay_ct.png (Clinical Dose Overlay) ---
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

    # --- 4. Challenge 4 Slice Assets (Metadata Stack - Whole Slices where valid) ---
    print("Generating Challenge 4 modality slices (Whole Context)...")
    # We use the 512x512 CT as the master slice
    master_ct = full_ct[:, :, 37]
    
    # For others, since they are 64x64, we will RESCALE them to 512x512 
    # to show the alignment in the stack.
    def rescale_and_crop(patch_3d):
        patch_2d = patch_3d[:, :, 32]
        # Rescale 64x64 to 512x512
        rescaled = zoom(patch_2d, (8, 8), order=3)
        return np.rot90(rescaled)

    def save_landscape_slice(img_2d, name, cmap, vmin=None, vmax=None, rot=0):
        if rot != 0:
            img_2d = rotate(img_2d, rot, reshape=False, order=3)
        plt.figure(figsize=(8, 3))
        # Crop wide landscape (central band)
        h, w = img_2d.shape
        wide_s = img_2d[int(h*0.35):int(h*0.65), :]
        plt.imshow(wide_s, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, name), dpi=200, bbox_inches='tight', transparent=True)
        plt.close()

    # CT Anatomy (Master 512x512)
    ct_master_2d = np.rot90(master_ct)
    save_landscape_slice(ct_master_2d, "ct_slice.png", "gray", -160, 240)
    save_landscape_slice(ct_master_2d, "ct_slice_rot.png", "gray", -160, 240, rot=45)
    
    # Dosemap (Rescaled UDE)
    dose_2d = rescale_and_crop(ude_patch)
    save_landscape_slice(dose_2d, "dosemap_slice.png", "hot")
    save_landscape_slice(dose_2d, "dosemap_slice_rot.png", "hot", rot=45)
    
    # SPECT AC (Rescaled MC Ground Truth)
    spect_ac_2d = rescale_and_crop(orig_patch)
    save_landscape_slice(spect_ac_2d, "spect_ac_slice.png", "magma")
    save_landscape_slice(spect_ac_2d, "spect_ac_slice_rot.png", "magma", rot=45)
    
    # SPECT NAC (Rescaled MC + Noise)
    spect_nac_2d = rescale_and_crop(orig_patch * (1.0 + 0.25 * np.random.randn(*dims)))
    save_landscape_slice(spect_nac_2d, "spect_nac_slice.png", "magma")
    save_landscape_slice(spect_nac_2d, "spect_nac_slice_rot.png", "magma", rot=45)

    # --- 5. Quantitative Lanes ---
    print("Generating dosimetry comparison lanes...")
    for data, name in [(cnn_patch, "dl_artifacts.png"), 
                       (approx_patch, "vsv_homo.png"), 
                       (ude_patch, "ude_highfi.png")]:
        plt.figure(figsize=(5, 6))
        # Use inferno for quantitative lanes as per guide
        plt.imshow(np.rot90(data[:, :, 32]), cmap='inferno', interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, name), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

    print(f"\nSUCCESS: All clinical assets generated in {out_dir}")

if __name__ == "__main__":
    generate_final_clinical_assets()
