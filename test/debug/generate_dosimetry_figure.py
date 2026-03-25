import os
import numpy as np
import matplotlib.pyplot as plt

def generate_dosimetry_fig_v3():
    vis_dir = "elsarticle/dosimetry/vis_results"
    pat_name = "FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_1__Pat47"
    pat_dir = os.path.join(vis_dir, pat_name)
    out_path = "elsarticle/figures_new/dosimetry_comparison.png"
    
    if not os.path.exists(pat_dir):
        dirs = [d for d in os.listdir(vis_dir) if os.path.isdir(os.path.join(vis_dir, d))]
        if not dirs: return
        pat_name = dirs[0]
        pat_dir = os.path.join(vis_dir, pat_name)

    with open(os.path.join(pat_dir, "dims.txt"), "r") as f:
        dims = tuple(map(int, f.read().strip().split(",")))
    
    ct = np.fromfile(os.path.join(pat_dir, "ct.bin"), dtype=np.float32).reshape(dims, order='F')
    pred = np.fromfile(os.path.join(pat_dir, "pred.bin"), dtype=np.float32).reshape(dims, order='F')
    orig = np.fromfile(os.path.join(pat_dir, "orig.bin"), dtype=np.float32).reshape(dims, order='F')
    approx = np.fromfile(os.path.join(pat_dir, "approx.bin"), dtype=np.float32).reshape(dims, order='F')
    
    cy = dims[1] // 2
    ct_s = np.rot90(ct[:, cy, :])
    approx_s = np.rot90(approx[:, cy, :])
    orig_s = np.rot90(orig[:, cy, :])
    pred_s = np.rot90(pred[:, cy, :])
    
    # IMPROVED WINDOWING: Use 98th percentile for better visibility
    vmax_dose = np.percentile(orig[orig > 0], 98.0)
    
    valid_idx = np.where(orig > 0.01 * vmax_dose)
    val_model = pred[valid_idx]
    val_mc = orig[valid_idx]
    val_model_scaled = val_model * (np.mean(val_mc) / (np.mean(val_model) + 1e-6))
    
    if len(val_mc) > 5000:
        idx = np.random.choice(len(val_mc), 5000, replace=False)
        val_model_scaled = val_model_scaled[idx]
        val_mc = val_mc[idx]
        
    mean_ba = (val_model_scaled + val_mc) / 2
    diff_ba = val_model_scaled - val_mc
    md = np.mean(diff_ba)
    sd = np.std(diff_ba)

    fig = plt.figure(figsize=(30, 6))
    gs = fig.add_gridspec(1, 5)
    
    def plot_slice(ax, data, title, cmap, vmax=None, vmin=0):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plot_slice(fig.add_subplot(gs[0, 0]), ct_s, "CT (Anatomy)", "gray", vmax=200, vmin=-150)
    plot_slice(fig.add_subplot(gs[0, 1]), orig_s, "Monte Carlo (Target)", "inferno", vmax=vmax_dose)
    plot_slice(fig.add_subplot(gs[0, 2]), pred_s, "No-Approx UDE (Ours)", "inferno", vmax=vmax_dose)
    plot_slice(fig.add_subplot(gs[0, 3]), approx_s, "Analytical Baseline", "inferno", vmax=vmax_dose)
    
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.scatter(mean_ba, diff_ba, alpha=0.4, s=3, color='forestgreen')
    ax5.axhline(md, color='red', linestyle='--', label=f'Mean Diff: {md:.1e}')
    ax5.axhline(md + 1.96*sd, color='blue', linestyle=':', label='±1.96 SD')
    ax5.axhline(md - 1.96*sd, color='blue', linestyle=':')
    ax5.set_xlabel("Average Dose (Relative)", fontsize=12)
    ax5.set_ylabel("Difference (Model - MC)", fontsize=12)
    ax5.set_title("Bland-Altman: UDE Accuracy", fontsize=16)
    ax5.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Brighter dosimetry figure saved to {out_path}")

if __name__ == "__main__":
    generate_dosimetry_fig_v3()
