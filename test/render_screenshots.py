#!/usr/bin/env python3
"""Render the synthetic-transforms grid from the visual_output NIfTI samples.

Each panel is drawn at its true physical field of view (size x spacing) on a
shared canvas, so size-changing operations (scale, crop, pad, resample) are
visibly different instead of being stretched to fill. Includes the fused-affine
panel (rotate + scale + translate in a single interpolation pass).

    julia +1.11 --startup-file=no --project=. test/generate_visual_samples.jl
    .venv/bin/python test/render_screenshots.py
"""
import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VIS = os.path.join(HERE, "visual_output")
OUT = os.path.join(VIS, "screenshots")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200,
                     "font.family": "DejaVu Sans", "savefig.bbox": "tight"})

# fused_transform_1 is the asymmetric block (rotation is visible); the sphere
# (…_2) would hide the rotation component of the fused transform.
SYNTH = [
    ("input_1.nii.gz",                "Input (block)"),
    ("rotated_0deg_1.nii.gz",         "Rotate 0°"),
    ("rotated_45deg_2.nii.gz",        "Rotate 45°"),
    ("translated_1.nii.gz",           "Translate +10 vox"),
    ("cropped_1.nii.gz",              "Crop 32³"),
    ("padded_1.nii.gz",               "Pad 74³"),
    ("sheared_xy_0.5_2.nii.gz",       "Shear XY 0.5"),
    ("scaled_0.5x_2.nii.gz",          "Scale 0.5×"),
    ("resample_spacing_2mm_1.nii.gz", "Resample 2 mm"),
    ("fused_transform_1.nii.gz",      "Fused affine\n(rot 30° · 0.8× · +5/−2)"),
]


def norm01(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        lo, hi = float(a.min()), float(max(a.max(), a.min() + 1e-6))
    return np.clip((a - lo) / (hi - lo), 0, 1)


def best_slice(a):
    t = a.min() + 0.1 * (a.max() - a.min() + 1e-9)
    return int(np.argmax([(a[i] > t).sum() for i in range(a.shape[0])]))


def load(path):
    im = sitk.ReadImage(path)
    a = sitk.GetArrayFromImage(im)          # (z, y, x)
    disp = np.rot90(norm01(a[best_slice(a)]))  # (x, y): rows<-x, cols<-y
    sx, sy, _ = im.GetSpacing()
    sz = im.GetSize()
    h = disp.shape[0] * sx                    # physical extent along rows (mm)
    w = disp.shape[1] * sy                    # physical extent along cols (mm)
    return disp, w, h, sz, (sx, sy)


def main():
    data, missing = [], []
    for fn, title in SYNTH:
        p = os.path.join(VIS, fn)
        if not os.path.exists(p):
            missing.append(fn)
            continue
        data.append((title, *load(p)))
    if not data:
        print("no inputs found in", VIS, file=sys.stderr)
        sys.exit(1)

    W = max(d[2] for d in data)
    H = max(d[3] for d in data)

    ncol = 5
    nrow = (len(data) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 3.0 * nrow),
                             constrained_layout=True)
    axes = axes.ravel()
    for ax in axes:
        ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])

    for ax, (title, disp, w, h, sz, sp) in zip(axes, data):
        ax.imshow(disp, cmap="gray", origin="upper", interpolation="nearest",
                  extent=[-w / 2, w / 2, -h / 2, h / 2], aspect="equal")
        ax.set_xlim(-W / 2, W / 2); ax.set_ylim(-H / 2, H / 2)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=3)
        ax.set_xlabel(f"{sz[0]}×{sz[1]}×{sz[2]}  ·  {sp[0]:.2g} mm", fontsize=7.5)

    for ax in axes[len(data):]:
        ax.set_visible(False)

    fig.suptitle("MedImages.jl batched transforms on synthetic phantoms "
                 "(panels drawn to physical scale)", fontsize=13, fontweight="bold")
    out = os.path.join(OUT, "Synth_transforms_grid.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)
    if missing:
        print("MISSING inputs:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
