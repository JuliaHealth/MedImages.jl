#!/usr/bin/env python3
"""Single grid: MedImages.jl vs SimpleITK across all spatial operations on a real CT.

For each operation the MedImages output defines the reference output geometry;
SimpleITK then resamples the ORIGINAL onto that exact geometry (identity transform
for crop/pad/scale/resample; the rotation transform for rotation). This compares the
two implementations' voxel values on an identical grid — no parameter guessing.
"""
import os, math
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans", "savefig.bbox": "tight"})

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_figures"))
os.makedirs(OUT, exist_ok=True)
ORIG = "/tmp/cmp_original.nii.gz"
AIR = -1000.0


def ctwin(a, lo=-160, hi=240):
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)


def sitk_ref(original, med_img, transform):
    """Resample original onto med_img's geometry with the given transform."""
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(med_img)
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(AIR)
    rs.SetTransform(transform)
    return rs.Execute(original)


def main():
    original = sitk.ReadImage(ORIG)
    c = original.GetSize()
    rot = sitk.Euler3DTransform()
    rot.SetCenter(original.TransformIndexToPhysicalPoint([c[0] // 2, c[1] // 2, c[2] // 2]))
    rot.SetRotation(0, 0, math.radians(-45.0))  # SimpleITK transform convention -> +45 CCW image

    # scale_mi(0.5) samples original at index 2*(i-1)+1 -> scale about the origin by 1/0.5=2
    scl = sitk.ScaleTransform(3, [2.0, 2.0, 1.0])
    scl.SetCenter(original.GetOrigin())

    ops = [
        ("Rotate 45°",      "/tmp/cmp_rotate.nii.gz",   rot),
        ("Scale 0.5×",      "/tmp/cmp_scale.nii.gz",     scl),
        ("Resample 2 mm",   "/tmp/cmp_resample.nii.gz",  sitk.Transform()),
        ("Crop 256×256×40", "/tmp/cmp_crop.nii.gz",      sitk.Transform()),
        ("Pad +40/+5",      "/tmp/cmp_pad.nii.gz",       sitk.Transform()),
    ]

    n = len(ops)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.4 * n), constrained_layout=True)
    for r, (name, path, T) in enumerate(ops):
        med = sitk.ReadImage(path)
        ref = sitk_ref(original, med, T)
        a_med = sitk.GetArrayFromImage(med)
        a_ref = sitk.GetArrayFromImage(ref)
        z = a_med.shape[0] // 2
        m, rf = ctwin(a_med[z]), ctwin(a_ref[z])
        diff = np.abs(m - rf)
        fg = (a_med > -900) | (a_ref > -900)
        pear = np.corrcoef(ctwin(a_med[fg]).ravel(), ctwin(a_ref[fg]).ravel())[0, 1]

        axes[r, 0].imshow(np.rot90(m), cmap="gray", aspect="equal", interpolation="bilinear")
        axes[r, 0].set_ylabel(name, fontsize=13, fontweight="bold")
        axes[r, 1].imshow(np.rot90(rf), cmap="gray", aspect="equal", interpolation="bilinear")
        im = axes[r, 2].imshow(np.rot90(diff), cmap="inferno", vmin=0, vmax=1, aspect="equal", interpolation="bilinear")
        axes[r, 2].set_title(f"|diff|   Pearson = {pear:.4f}", fontsize=11)
        for c2 in range(3):
            axes[r, c2].set_xticks([]); axes[r, c2].set_yticks([])
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.03)
        print(f"{name:18s} pearson={pear:.4f}  med{a_med.shape} ref{a_ref.shape}")

    axes[0, 0].set_title("MedImages.jl", fontsize=14, fontweight="bold")
    axes[0, 1].set_title("SimpleITK (same grid)", fontsize=14, fontweight="bold")
    fig.suptitle("MedImages.jl vs SimpleITK across spatial operations (real CT, axial mid-slice, soft-tissue window)",
                 fontsize=15, fontweight="bold")
    out = os.path.join(OUT, "fig14_medimages_vs_simpleitk_allops.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
