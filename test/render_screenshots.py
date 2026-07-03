#!/usr/bin/env python3
"""Render labeled screenshot PNGs from the visual_output NIfTI samples.

Reproduces the per-transform screenshots + comparison grids that were
previously generated against the JuliaHealth MedImages.jl clone, now from
the outputs of Jakub Mitura's fork.
"""
import os
import sys
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
VIS = os.path.join(HERE, "visual_output")
OUT = os.path.join(VIS, "screenshots")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SZ = 420
os.makedirs(OUT, exist_ok=True)


def norm01(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        lo, hi = a.min(), max(a.max(), a.min() + 1e-6)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def ct_window(a, lo=-160, hi=240):
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)


def mid_slice(path, window=None):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    if window == "ct":
        z = arr.shape[0] // 2
    else:
        # pick the slice with the most foreground so off-center content stays visible
        thresh = arr.min() + 0.1 * (arr.max() - arr.min() + 1e-9)
        z = int(np.argmax([(arr[i] > thresh).sum() for i in range(arr.shape[0])]))
    sl = arr[z]
    sl = ct_window(sl) if window == "ct" else norm01(sl)
    return sl, img


def panel(sl, text, sub=""):
    img = Image.fromarray((np.rot90(sl) * 255).astype(np.uint8)).resize(
        (SZ, SZ), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(img)
    try:
        ft = ImageFont.truetype(FONT, 24)
        fs = ImageFont.truetype(FONT, 16)
    except Exception:
        ft = fs = ImageFont.load_default()
    d.rectangle([0, 0, SZ, 64 if sub else 40], fill=(20, 20, 20))
    d.text((8, 8), text, fill=(255, 255, 255), font=ft)
    if sub:
        d.text((8, 38), sub, fill=(180, 180, 180), font=fs)
    return img


def meta_str(img):
    sp = img.GetSpacing()
    sz = img.GetSize()
    return f"size {sz[0]}x{sz[1]}x{sz[2]}  spacing {sp[0]:.2g}/{sp[1]:.2g}/{sp[2]:.2g} mm"


SYNTH = [
    ("input_1.nii.gz",              "1. Input (block)"),
    ("rotated_0deg_1.nii.gz",       "2. Rotate 0deg"),
    ("rotated_45deg_2.nii.gz",      "3. Rotate 45deg"),
    ("translated_1.nii.gz",         "4. Translate +10x"),
    ("cropped_1.nii.gz",            "5. Crop 32^3"),
    ("padded_1.nii.gz",             "6. Pad 74^3"),
    ("sheared_xy_0.5_2.nii.gz",     "7. Shear XY 0.5"),
    ("scaled_0.5x_2.nii.gz",        "8. Scale 0.5x"),
    ("resample_spacing_2mm_1.nii.gz", "9. Resample 2mm"),
]


def main():
    panels = []
    missing = []
    for i, (fname, label) in enumerate(SYNTH, 1):
        path = os.path.join(VIS, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        sl, img = mid_slice(path)
        p = panel(sl, label, meta_str(img))
        out_name = f"{i}_{label.split('. ', 1)[1].lower().replace(' ', '_').replace('^', '').replace('+', '').replace('.', '')}.png"
        p.save(os.path.join(OUT, out_name))
        panels.append(np.array(p))
        print("wrote", out_name)

    if panels:
        rows = []
        for r in range(0, len(panels), 3):
            row = panels[r:r + 3]
            while len(row) < 3:
                row.append(np.full_like(panels[0], 15))
            rows.append(np.hstack(row))
        grid = np.vstack(rows)
        Image.fromarray(grid).save(os.path.join(OUT, "Synth_transforms_grid.png"))
        print("wrote Synth_transforms_grid.png")

    if missing:
        print("MISSING inputs:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)


CT = [
    ("/tmp/real_ct_original.nii.gz",      "CT 1. Original"),
    ("/tmp/real_ct_rotated_45.nii.gz",    "CT 2. Rotated 45deg"),
    ("/tmp/real_ct_scaled_half.nii.gz",   "CT 3. Scaled 0.5x"),
    ("/tmp/real_ct_resampled_2mm.nii.gz", "CT 4. Resampled 2mm"),
]


def main_ct():
    panels = []
    for i, (path, label) in enumerate(CT, 1):
        if not os.path.exists(path):
            print("MISSING:", path, file=sys.stderr)
            continue
        sl, img = mid_slice(path, window="ct")
        p = panel(sl, label, meta_str(img))
        out_name = f"CT_{i}_{label.split('. ', 1)[1].lower().replace(' ', '_').replace('.', '')}.png"
        p.save(os.path.join(OUT, out_name))
        panels.append(np.array(p))
        print("wrote", out_name)

    if len(panels) == 4:
        grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        Image.fromarray(grid).save(os.path.join(OUT, "CT_comparison_grid_labeled.png"))
        print("wrote CT_comparison_grid_labeled.png")


if __name__ == "__main__":
    main()
    main_ct()
