import os
import numpy as np
import matplotlib.pyplot as plt

def generate_final_vis_enhanced():
    vis_dir = "elsarticle/dosimetry/vis_results_64"
    pat_name = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat47"
    pat_dir = os.path.join(vis_dir, pat_name)
    out_path = "elsarticle/figures_new/dosimetry_comparison_all.png"
    
    dims = (64, 64, 64)
    def load_bin(f):
        return np.fromfile(os.path.join(pat_dir, f), dtype=np.float32).reshape(dims, order='F')

    # Load top 4 + 4 representative baselines
    panels = [
        (load_bin("ct.bin"), "gray", "(A) Anatomy (CT)", -160, 240, "black"),
        (load_bin("orig.bin"), "inferno", "(B) Monte Carlo (GT)", None, None, "white"),
        (load_bin("ude.bin"), "inferno", "(C) UDE Champion", None, None, "white"),
        (load_bin("cnn.bin"), "inferno", "(D) Stabilized CNN", None, None, "white"),
        (load_bin("approx.bin"), "inferno", "(E) VSV Baseline", None, None, "white"),
        (load_bin("pbpk.bin"), "inferno", "(F) PBPK-ML", None, None, "white"),
        (load_bin("dblur.bin"), "inferno", "(G) DblurDoseNet", None, None, "white"),
        (load_bin("semi.bin"), "inferno", "(H) SemiDose", None, None, "white")
    ]
    
    # 2 rows, 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    plt.subplots_adjust(wspace=0.02, hspace=0.1)
    
    cz = 32
    axes_flat = axes.flatten()

    for i, (data, cmap, title, vmin, vmax, text_color) in enumerate(panels):
        ax = axes_flat[i]
        slice_data = np.rot90(data[:, :, cz])
        
        if vmin is None: vmin = 0
        if vmax is None:
            valid = slice_data[slice_data > 0.01 * np.max(slice_data)]
            vmax = np.percentile(valid, 98.0) if len(valid) > 0 else np.max(slice_data)

        im = ax.imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax.set_title(title, fontsize=22, fontweight='bold', pad=10)
        ax.axis('off')

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Enhanced 2-row figure saved to {out_path}")

if __name__ == "__main__":
    generate_final_vis_enhanced()
