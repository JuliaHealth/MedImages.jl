import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

def generate_assets():
    vis_dir = "elsarticle/dosimetry/vis_results_64"
    pat_name = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat47"
    pat_dir = os.path.join(vis_dir, pat_name)
    out_dir = "elsarticle/figures_new/clinical_assets"
    os.makedirs(out_dir, exist_ok=True)
    
    dims = (64, 64, 64)
    def load_bin(f):
        return np.fromfile(os.path.join(pat_dir, f), dtype=np.float32).reshape(dims, order='F')

    ct = load_bin("ct.bin")
    ude = load_bin("ude.bin")
    orig = load_bin("orig.bin")
    approx = load_bin("approx.bin")
    cnn = load_bin("cnn.bin")
    
    cz = 32
    
    # 1. mip_wholebody.png (MIP of UDE dose, but the guide says "high-contrast, black-and-white")
    mip = np.max(ude, axis=1) # MIP along Y-axis
    plt.figure(figsize=(4, 5))
    plt.imshow(np.rot90(mip), cmap='gray_r', interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "mip_wholebody.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    # 2. resampling_ct.png (axial CT cross-section showing spatial resampling/rotation)
    ct_slice = np.rot90(ct[:, :, cz])
    ct_rot = rotate(ct_slice, 45, reshape=False, order=3)
    plt.figure(figsize=(4, 4))
    plt.imshow(ct_rot, cmap='gray', vmin=-160, vmax=240, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "resampling_ct.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    # 3. dose_overlay_ct.png (UDE dose map overlay on CT)
    plt.figure(figsize=(4, 5))
    plt.imshow(np.rot90(ct[:, :, cz]), cmap='gray', vmin=-160, vmax=240)
    dose_mask = np.rot90(ude[:, :, cz])
    dose_mask = np.ma.masked_where(dose_mask < 0.05 * np.max(dose_mask), dose_mask)
    plt.imshow(dose_mask, cmap='hot', alpha=0.6, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "dose_overlay_ct.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    # 4. Challenge 4 Slice Assets (The 3D Metadata Stack)
    def save_full_slice(data, name, cmap, vmin=None, vmax=None, alpha=1.0):
        plt.figure(figsize=(5, 5))
        s = np.rot90(data[:, :, cz])
        plt.imshow(s, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches='tight', transparent=True)
        plt.close()

    save_full_slice(ct, "ct_slice.png", "gray", -160, 240)
    save_full_slice(ude, "dosemap_slice.png", "hot")
    save_full_slice(orig, "spect_ac_slice.png", "magma")
    save_full_slice(orig * (1.0 + 0.3 * np.random.randn(*dims)), "spect_nac_slice.png", "magma")

    def save_full_slice_rot(data, name, cmap, vmin=None, vmax=None):
        plt.figure(figsize=(5, 5))
        s = np.rot90(data[:, :, cz])
        s_rot = rotate(s, 45, reshape=False, order=3)
        plt.imshow(s_rot, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches='tight', transparent=True)
        plt.close()

    save_full_slice_rot(ct, "ct_slice_rot.png", "gray", -160, 240)
    save_full_slice_rot(ude, "dosemap_slice_rot.png", "hot")
    save_full_slice_rot(orig, "spect_ac_slice_rot.png", "magma")
    save_full_slice_rot(orig * (1.0 + 0.3 * np.random.randn(*dims)), "spect_nac_slice_rot.png", "magma")
    
    # 5. Quantitative UDE Dosimetry Experiment Assets (For the 3 Comparison Lanes)
    # dl_artifacts.png (Using CNN as DL model example)
    plt.figure(figsize=(4, 5))
    plt.imshow(np.rot90(cnn[:, :, cz]), cmap='inferno', interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "dl_artifacts.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    # vsv_homo.png (Analytical Baseline)
    plt.figure(figsize=(4, 5))
    plt.imshow(np.rot90(approx[:, :, cz]), cmap='inferno', interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "vsv_homo.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    # ude_highfi.png (UDE Champion)
    plt.figure(figsize=(4, 5))
    plt.imshow(np.rot90(ude[:, :, cz]), cmap='inferno', interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, "ude_highfi.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    print(f"Clinical assets generated in {out_dir}")

if __name__ == "__main__":
    generate_assets()
