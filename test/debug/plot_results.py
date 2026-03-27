import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

vis_dir = "elsarticle/dosimetry/vis_results"
if not os.path.exists(vis_dir):
    print(f"Directory {vis_dir} does not exist. Run Julia script first.")
    exit()

def process_patient(pat_name):
    pat_dir = os.path.join(vis_dir, pat_name)
    dims_file = os.path.join(pat_dir, "dims.txt")
    if not os.path.exists(dims_file):
        return
        
    with open(dims_file, "r") as f:
        dims = tuple(map(int, f.read().strip().split(",")))
    
    ct = np.fromfile(os.path.join(pat_dir, "ct.bin"), dtype=np.float32).reshape(dims, order='F')
    pred = np.fromfile(os.path.join(pat_dir, "pred.bin"), dtype=np.float32).reshape(dims, order='F')
    orig = np.fromfile(os.path.join(pat_dir, "orig.bin"), dtype=np.float32).reshape(dims, order='F')
    approx = np.fromfile(os.path.join(pat_dir, "approx.bin"), dtype=np.float32).reshape(dims, order='F')
    
    # Find slice with maximum dose sum
    cz = np.argmax(np.sum(orig, axis=(0, 1)))
    cy = np.argmax(np.sum(orig, axis=(0, 2)))
    cx = np.argmax(np.sum(orig, axis=(1, 2)))
    
    planes = [
        ("Transverse", np.rot90(ct[:, :, cz]), np.rot90(approx[:, :, cz]), np.rot90(orig[:, :, cz]), np.rot90(pred[:, :, cz])),
        ("Coronal", np.rot90(ct[:, cy, :]), np.rot90(approx[:, cy, :]), np.rot90(orig[:, cy, :]), np.rot90(pred[:, cy, :])),
        ("Sagittal", np.rot90(ct[cx, :, :]), np.rot90(approx[cx, :, :]), np.rot90(orig[cx, :, :]), np.rot90(pred[cx, :, :]))
    ]
    
    ct_vmin, ct_vmax = -150, 250
    
    for plane_name, ct_s, approx_s, orig_s, pred_s in planes:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"{pat_name} - {plane_name} (Independent Scaling)", fontsize=16)
        
        def plot_slice(ax, data, title, cmap, vmin=None, vmax=None):
            if vmin is None: vmin = 0
            if vmax is None:
                valid = data[data > 0]
                vmax = np.percentile(valid, 99.5) if len(valid) > 0 else (np.max(data) if np.max(data) > 0 else 1.0)
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{title}\nMax: {np.max(data):.2e}")
            fig.colorbar(im, ax=ax)
            ax.axis('off')

        plot_slice(axes[0], ct_s, "CT", "bone", vmin=ct_vmin, vmax=ct_vmax)
        plot_slice(axes[1], approx_s, "Baseline Approx", "inferno")
        plot_slice(axes[2], orig_s, "MC Target", "inferno")
        plot_slice(axes[3], pred_s, "Predicted (Model)", "inferno")
        
        plt.tight_layout()
        filename = f"{pat_name}_{plane_name.lower()}.png"
        plt.savefig(os.path.join(pat_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)

for d in os.listdir(vis_dir):
    if os.path.isdir(os.path.join(vis_dir, d)):
        process_patient(d)

print("Visualizations generated with independent scaling.")
