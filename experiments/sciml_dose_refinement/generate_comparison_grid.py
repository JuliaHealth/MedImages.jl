import os
import numpy as np
import matplotlib.pyplot as plt

def load_bin(path, shape=(64, 64, 64)):
    return np.fromfile(path, dtype=np.float32).reshape(shape)

def get_mip(data):
    # Coronal MIP (along axis 1)
    mip = np.max(data, axis=1)
    return np.rot90(mip)

def run():
    pat_out_dir = "/home/user/MedImages.jl/val_outputs/FDM_DPI-2024-7-KRN_Lu177_PSMA__SPECT_Tc_0__Pat44"
    
    # Load data
    ct = load_bin(os.path.join(pat_out_dir, "ct.bin"))
    mc = load_bin(os.path.join(pat_out_dir, "mc.bin"))
    baseline = load_bin(os.path.join(pat_out_dir, "baseline.bin"))
    
    print(f"MC: min={mc.min():.4f}, max={mc.max():.4f}, mean={mc.mean():.4f}")
    print(f"Baseline: min={baseline.min():.4f}, max={baseline.max():.4f}, mean={baseline.mean():.4f}")

    models = [
        ("UDE Improved", load_bin(os.path.join(pat_out_dir, "ude_imp.bin"))),
        ("UDE No-Approx", load_bin(os.path.join(pat_out_dir, "ude_noapp.bin"))),
        ("CNN Improved", load_bin(os.path.join(pat_out_dir, "cnn_imp.bin"))),
        ("CNN+Approx (Hybrid)", load_bin(os.path.join(pat_out_dir, "cnn_app.bin"))),
        ("Pure CNN", load_bin(os.path.join(pat_out_dir, "pure_cnn.bin"))),
        ("Analytical Baseline", baseline)
    ]
    
    for name, pred in models:
        print(f"{name}: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
    
    # Grid: 7 rows x 3 columns
    fig, axes = plt.subplots(7, 3, figsize=(12, 24))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    # Row 1: CT | MC | Baseline
    axes[0, 0].imshow(np.rot90(ct[:, 32, :]), cmap='gray')
    axes[0, 0].set_title("CT Context (Slice)")
    
    vmax_mc = np.percentile(mc, 99.5)
    if vmax_mc == 0: vmax_mc = 1.0
    axes[0, 1].imshow(get_mip(mc), cmap='hot', vmin=0, vmax=vmax_mc)
    axes[0, 1].set_title("Monte Carlo (GT) MIP")
    
    # Scale baseline for visualization in row 1
    b_abs = np.abs(baseline); b_max = np.percentile(b_abs, 99.5)
    b_scaled = b_abs * (vmax_mc / b_max) if b_max > 1e-8 else b_abs
    axes[0, 2].imshow(get_mip(b_scaled), cmap='hot', vmin=0, vmax=vmax_mc)
    axes[0, 2].set_title("Analytical Baseline MIP (Scaled)")
    
    for i in range(3): axes[0, i].axis('off')

    # Rows 2-7: Model | Subtraction | GT
    for idx, (name, pred) in enumerate(models):
        r = idx + 1
        
        # Auto-scale pred to MC range for fair visual comparison
        p_abs = np.abs(pred)
        p_max = np.percentile(p_abs, 99.5)
        p_scaled = p_abs * (vmax_mc / p_max) if p_max > 1e-8 else p_abs
            
        # Column 0: Model Prediction
        axes[r, 0].imshow(get_mip(p_scaled), cmap='hot', vmin=0, vmax=vmax_mc)
        axes[r, 0].set_title(f"{name} MIP (Scaled)")
        
        # Column 1: Subtraction (MC - Model)
        sub = mc - p_scaled
        
        # MASK: Force areas with very low MC dose to zero residual (white background)
        mask = mc < (0.01 * vmax_mc)
        sub[mask] = 0.0
        
        vmax_sub = np.percentile(np.abs(sub), 99.5)
        if vmax_sub == 0: vmax_sub = 1.0
        
        im = axes[r, 1].imshow(get_mip(sub), cmap='coolwarm', vmin=-vmax_sub, vmax=vmax_sub)
        axes[r, 1].set_title(f"Residual: MC - {name}")
        plt.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.04)

        # Column 2: MC (GT) for reference
        axes[r, 2].imshow(get_mip(mc), cmap='hot', vmin=0, vmax=vmax_mc)
        axes[r, 2].set_title("Monte Carlo (GT)")

        for i in range(3): axes[r, i].axis('off')

    plt.suptitle("Comprehensive Model Comparison & Error Residuals (Pat44)", fontsize=20, y=0.95)
    out_path = "/home/user/MedImages.jl/val_outputs/comprehensive_comparison_grid.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Comparison grid saved to {out_path}")

if __name__ == "__main__":
    run()
