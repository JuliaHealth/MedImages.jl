#!/usr/bin/env python3
"""Render the real-CT transforms grid from the NIfTI produced by
experiments/make_real_ct_samples.jl.

Unlike a naive per-panel imshow (which stretches every image to fill its axes
and hides size changes), each panel is drawn at its true physical field of view
(size x spacing) on a shared canvas. A 0.5x scale therefore appears half-size,
a crop appears smaller, and a pad appears larger -- the size change is visible.

    .venv/bin/python experiments/render_real_ct_grid.py
"""
import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "test", "visual_output", "real_ct")
OUT = os.path.join(ROOT, "test", "visual_output", "screenshots", "CT_comparison_grid_labeled.png")

plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200,
                     "font.family": "DejaVu Sans", "savefig.bbox": "tight"})

PANELS = [
    ("ct_original.nii.gz",   "Original"),
    ("ct_rotate45.nii.gz",   "Rotate 45°"),
    ("ct_scale05.nii.gz",    "Scale 0.5×"),
    ("ct_resample2mm.nii.gz", "Resample 2 mm"),
    ("ct_crop.nii.gz",       "Crop 256×256×40"),
    ("ct_pad.nii.gz",        "Pad +40/+5"),
    ("ct_fused.nii.gz",      "Fused affine\n(rot 30° · 0.8× · +5/−2)"),
]


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
    return disp, w, h, sz, (sx, sy)


def main():
    data = []
    for fn, title in PANELS:
        p = os.path.join(SRC, fn)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}; run make_real_ct_samples.jl first")
        data.append((title, *load(p)))

    W = max(d[2] for d in data)              # shared canvas width (mm)
    H = max(d[3] for d in data)              # shared canvas height (mm)

    ncol = 4
    nrow = (len(data) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.4 * nrow),
                             constrained_layout=True)
    axes = axes.ravel()
    for ax in axes:
        ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])

    for ax, (title, disp, w, h, sz, sp) in zip(axes, data):
        ax.imshow(disp, cmap="gray", origin="upper", interpolation="bilinear",
                  extent=[-w / 2, w / 2, -h / 2, h / 2], aspect="equal")
        ax.set_xlim(-W / 2, W / 2); ax.set_ylim(-H / 2, H / 2)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
        ax.set_xlabel(f"{sz[0]}×{sz[1]}×{sz[2]}  ·  {sp[0]:.2g}/{sp[1]:.2g} mm  ·  "
                      f"FOV {w:.0f}×{h:.0f} mm", fontsize=7.5)

    for ax in axes[len(data):]:
        ax.set_visible(False)

    fig.suptitle("MedImages.jl spatial operations on a real CT volume "
                 "(axial mid-slice, soft-tissue window, panels drawn to physical scale)",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
