import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def load_sagittal_slice_with_physics(path):
    img = nib.load(path)
    data = img.get_fdata()
    zooms = img.header.get_zooms()[:3]
    
    # Sagittal slice (central X)
    x_idx = data.shape[0] // 2
    slice_2d = data[x_idx, :, :]
    
    # Physical Dimensions for Sagittal (Y-Z)
    width = data.shape[1] * zooms[1]
    height = data.shape[2] * zooms[2]
    
    # Rotated for standard sagittal view (Head up)
    return np.rot90(slice_2d), [0, width, 0, height]

def window_image(img, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    return np.clip(img, img_min, img_max)

def generate_figure_sagittal():
    pre_dir = "test_data/doses/pre"
    post_dir = "test_data/doses/post"
    out_path = "elsarticle/figures_new/transformation_consistency.png"
    
    print("Extracting and ISO-resampling SAGITTAL slices (for X-rotation)...")
    # Pre
    s_ct_pre, ext_pre = load_sagittal_slice_with_physics(os.path.join(pre_dir, "CT.nii.gz"))
    s_dose_pre, _ = load_sagittal_slice_with_physics(os.path.join(pre_dir, "Dosemap.nii.gz"))
    s_mask_pre, _ = load_sagittal_slice_with_physics(os.path.join(pre_dir, "liver.nii.gz"))
    
    # Post
    s_ct_post, ext_post = load_sagittal_slice_with_physics(os.path.join(post_dir, "CT.nii.gz"))
    s_dose_post, _ = load_sagittal_slice_with_physics(os.path.join(post_dir, "Dosemap.nii.gz"))
    s_mask_post, _ = load_sagittal_slice_with_physics(os.path.join(post_dir, "liver.nii.gz"))
    
    # Windowing
    s_ct_pre_w = window_image(s_ct_pre, 40, 400)
    s_ct_post_w = window_image(s_ct_post, 40, 400)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 20))
    plt.subplots_adjust(wspace=0.01, hspace=0.05)
    
    def plot_panel(ax, ct, overlay, extent, title, cmap_over, label):
        ax.imshow(ct, cmap='gray', vmin=-160, vmax=240, extent=extent, aspect='equal')
        if overlay is not None:
            ax.imshow(overlay, cmap=cmap_over, alpha=0.45, extent=extent, aspect='equal')
        ax.set_title(f"({label}) {title}", fontsize=22, fontweight='bold', color='black')
        ax.axis('off')

    plot_panel(axes[0, 0], s_ct_pre_w, s_mask_pre, ext_pre, "Original: CT + Mask", "cool", "A")
    plot_panel(axes[0, 1], s_ct_pre_w, s_dose_pre, ext_pre, "Original: CT + Dose", "inferno", "B")
    plot_panel(axes[1, 0], s_ct_post_w, s_mask_post, ext_post, "Transformed: CT + Mask", "cool", "C")
    plot_panel(axes[1, 1], s_ct_post_w, s_dose_post, ext_post, "Transformed: CT + Dose", "inferno", "D")
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Sagittal-view consistency figure saved to {out_path}")

if __name__ == "__main__":
    generate_figure_sagittal()
