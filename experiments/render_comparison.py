import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/workspaces/MedImages.jl"
SRC = os.path.join(ROOT, "test", "visual_output", "comparisons")
OUT = os.path.join(ROOT, "test", "visual_output", "screenshots", "SITK_vs_MedImages_ArbitraryRotation.png")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200,
                     "font.family": "DejaVu Sans", "savefig.bbox": "tight"})

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
    sitk_path = os.path.join(SRC, "sitk_rotated30.nii.gz")
    medimages_path = os.path.join(SRC, "medimages_rotated30.nii.gz")
    orig_path = os.path.join(ROOT, "test_data", "volume-0.nii.gz")
    
    orig_disp, orig_w, orig_h, orig_sz, orig_sp, _ = load(orig_path)
    sitk_disp, w, h, sz, sp, sitk_a = load(sitk_path)
    med_disp, _, _, _, _, med_a = load(medimages_path)
    
    diff_a = np.abs(sitk_a.astype(np.float32) - med_a.astype(np.float32))
    diff_disp = np.rot90(np.clip(diff_a[diff_a.shape[0]//2] / 100.0, 0, 1)) # Normalize diff for visibility

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    
    for ax in axes:
        ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])

    # Original
    axes[0].imshow(orig_disp, cmap="gray", origin="upper", interpolation="bilinear",
                  extent=[-orig_w / 2, orig_w / 2, -orig_h / 2, orig_h / 2], aspect="equal")
    axes[0].set_title("Original CT", fontsize=11, fontweight="bold", pad=4)
    
    # SITK
    axes[1].imshow(sitk_disp, cmap="gray", origin="upper", interpolation="bilinear",
                  extent=[-w / 2, w / 2, -h / 2, h / 2], aspect="equal")
    axes[1].set_title("SimpleITK (30° Rot + Resample)", fontsize=11, fontweight="bold", pad=4)
    
    # MedImages
    axes[2].imshow(med_disp, cmap="gray", origin="upper", interpolation="bilinear",
                  extent=[-w / 2, w / 2, -h / 2, h / 2], aspect="equal")
    axes[2].set_title("MedImages.jl (30° Rot + Resample)", fontsize=11, fontweight="bold", pad=4)
    
    # Diff
    axes[3].imshow(diff_disp, cmap="inferno", origin="upper", interpolation="bilinear",
                  extent=[-w / 2, w / 2, -h / 2, h / 2], aspect="equal")
    axes[3].set_title("Absolute Difference (x100 intensity)", fontsize=11, fontweight="bold", pad=4)

    fig.suptitle("Comparison: Arbitrary Affine Matrix Resampling (SimpleITK vs MedImages.jl)",
                 fontsize=14, fontweight="bold")
    fig.savefig(OUT)
    plt.close(fig)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
