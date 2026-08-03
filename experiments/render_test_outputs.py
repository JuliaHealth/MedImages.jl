import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/jm/.gemini/antigravity-cli/brain/853d9462-f5ad-4cf0-874c-6c7072725016/SITK_vs_MedImages.png"
# Write to host path directly? No, write to workspace then copy!
OUT_WORKSPACE = "/workspaces/MedImages.jl/SITK_vs_MedImages.png"

def ctwin(a, lo=-160, hi=240):
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)

def load(path):
    im = sitk.ReadImage(path)
    a = sitk.GetArrayFromImage(im)          # (z, y, x)
    z = a.shape[0] // 2
    disp = np.rot90(ctwin(a[z]))            # (x, y): rows<-x, cols<-y
    sx, sy, _ = im.GetSpacing()
    sz = im.GetSize()
    h = disp.shape[0] * sx                   # physical extent along rows (mm)
    w = disp.shape[1] * sy                   # physical extent along cols (mm)
    return disp, w, h, sz, (sx, sy), a

def main():
    sitk_path = "/workspaces/MedImages.jl/test/resample_to_target_tests/outputs/resample_to_image/20260803_075631_6878/resampled_Linear_sitk.nii.gz"
    medimages_path = "/workspaces/MedImages.jl/test/resample_to_target_tests/outputs/resample_to_image/20260803_075631_6878/resampled_Linear_mi.nii.gz"
    
    sitk_disp, w, h, sz, sp, sitk_a = load(sitk_path)
    med_disp, _, _, _, _, med_a = load(medimages_path)
    
    diff_a = np.abs(sitk_a.astype(np.float32) - med_a.astype(np.float32))
    diff_disp = np.rot90(np.clip(diff_a[diff_a.shape[0]//2] / 100.0, 0, 1))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    
    for ax in axes:
        ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])

    # SITK
    axes[0].imshow(sitk_disp, cmap="gray", origin="upper", interpolation="bilinear",
                  extent=[-w / 2, w / 2, -h / 2, h / 2], aspect="equal")
    axes[0].set_title("SimpleITK Linear", fontsize=11, fontweight="bold", pad=4)
    
    # MedImages
    axes[1].imshow(med_disp, cmap="gray", origin="upper", interpolation="bilinear",
                  extent=[-w / 2, w / 2, -h / 2, h / 2], aspect="equal")
    axes[1].set_title("MedImages.jl Affine", fontsize=11, fontweight="bold", pad=4)
    
    # Diff
    axes[2].imshow(diff_disp, cmap="inferno", origin="upper", interpolation="bilinear",
                  extent=[-w / 2, w / 2, -h / 2, h / 2], aspect="equal")
    axes[2].set_title("Absolute Difference", fontsize=11, fontweight="bold", pad=4)

    fig.suptitle("Comparison: MedImages vs SimpleITK Resampling",
                 fontsize=14, fontweight="bold")
    fig.savefig(OUT_WORKSPACE)
    plt.close(fig)
    print("wrote", OUT_WORKSPACE)

if __name__ == "__main__":
    main()
