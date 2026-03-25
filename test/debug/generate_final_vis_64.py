import os
import numpy as np
import matplotlib.pyplot as plt

def generate_final_vis_64_fixed():
    vis_dir = "elsarticle/dosimetry/vis_results_64"
    pat_name = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat47"
    pat_dir = os.path.join(vis_dir, pat_name)
    out_path = "elsarticle/figures_new/dosimetry_comparison_64.png"
    
    if not os.path.exists(pat_dir):
        print(f"Directory {pat_dir} not found.")
        return

    dims = (64, 64, 64)
    def load_bin(f):
        return np.fromfile(os.path.join(pat_dir, f), dtype=np.float32).reshape(dims, order='F')

    ct = load_bin("ct.bin")
    ude = load_bin("ude.bin")
    cnn = load_bin("cnn.bin")
    orig = load_bin("orig.bin")
    approx = load_bin("approx.bin")
    
    cz = 32
    slices = [
        (np.rot90(ct[:, :, cz]), "gray", "(A) Anatomy (CT)", -160, 240, "black"),
        (np.rot90(orig[:, :, cz]), "inferno", "(B) Monte Carlo (GT)", None, None, "white"),
        (np.rot90(ude[:, :, cz]), "inferno", "(C) No-Approx UDE", None, None, "white"),
        (np.rot90(cnn[:, :, cz]), "inferno", "(D) Stabilized CNN", None, None, "white"),
        (np.rot90(approx[:, :, cz]), "inferno", "(E) Baseline Approx", None, None, "white")
    ]
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 6)
    
    row1_axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]
    row2_axes = [fig.add_subplot(gs[1, 1:3]), fig.add_subplot(gs[1, 3:5])]
    all_axes = row1_axes + row2_axes
    plt.subplots_adjust(wspace=0.01, hspace=0.05)

    for i, (data, cmap, title, vmin, vmax, text_color) in enumerate(slices):
        ax = all_axes[i]
        if vmin is None: vmin = 0
        if vmax is None:
            valid = data[data > 0.01 * np.max(data)] if np.max(data) > 0 else []
            vmax = np.percentile(valid, 98.0) if len(valid) > 0 else np.max(data)
            if vmax <= vmin: vmax = vmin + 1e-6

        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
        # VISUAL EXPERT: Explicit text color for visibility
        ax.set_title(title, fontsize=22, fontweight='bold', y=0.92, color=text_color)
        ax.axis('off')

    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fixed dosimetry figure saved to {out_path}")

if __name__ == "__main__":
    generate_final_vis_64_fixed()
