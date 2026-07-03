#!/usr/bin/env python3
"""Compare MedImages local-deposition dose against the F-18 DPK reference dose.

Produces voxel metrics and a labeled comparison figure into the screenshots dir.
"""
import os
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", "tcia_pet")
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "..", "test", "visual_output", "screenshots")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
os.makedirs(OUTDIR, exist_ok=True)


def load(name):
    return sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(BASE, name)))  # (z,y,x)


def metrics(local, dpk, m, label):
    l = local[m]; d = dpk[m]
    pear = np.corrcoef(l, d)[0, 1]
    ln = local / np.percentile(local[local > 0], 99)
    dn = dpk / np.percentile(dpk[dpk > 0], 99)
    mae = np.mean(np.abs(ln[m] - dn[m]))
    print(f"[{label}] voxels={m.sum():,}  Pearson={pear:.4f}  "
          f"normMAE={mae:.4f}  mean local/DPK={np.mean(l)/np.mean(d):.4f}")
    return pear


def main():
    local = load("dose_local.nii.gz")
    dpk = load("dose_dpk.nii.gz")
    act = load("activity.nii.gz")
    den = load("density.nii.gz")

    # body mask: exclude air (density floor 0.034). Keep soft tissue + lung.
    body = den > 0.15
    fg = ((local > 0) | (dpk > 0))
    print("=== whole FOV (includes air-boundary amplification) ===")
    metrics(local, dpk, fg, "all-fg")
    print("=== body-masked (density>0.15, clinically meaningful) ===")
    pear = metrics(local, dpk, fg & body, "body")

    # representative mid-body axial slice: most body-masked activity,
    # excluding the outer 15% of slices (bladder/edge artifacts)
    actm = act * body
    z0, z1 = int(0.15 * act.shape[0]), int(0.85 * act.shape[0])
    sums = [actm[z].sum() if z0 <= z <= z1 else -1 for z in range(act.shape[0])]
    Z = int(np.argmax(sums))
    # zero out air for display so windowing reflects tissue dose
    local = local * body
    dpk = dpk * body
    SZ = 360

    def panel(a2d, text, vmax=None, cmap_hot=False):
        v = a2d.astype(np.float32)
        if vmax is None:
            vmax = np.percentile(v[v > 0], 99) if (v > 0).any() else 1.0
        n = np.clip(v / (vmax + 1e-9), 0, 1)
        n = np.rot90(n)
        if cmap_hot:
            r = np.clip(n * 3, 0, 1); g = np.clip(n * 3 - 1, 0, 1); b = np.clip(n * 3 - 2, 0, 1)
            rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
        else:
            rgb = (np.stack([n] * 3, -1) * 255).astype(np.uint8)
        img = Image.fromarray(rgb).resize((SZ, SZ), Image.BILINEAR)
        dd = ImageDraw.Draw(img)
        try:
            ft = ImageFont.truetype(FONT, 18)
        except Exception:
            ft = ImageFont.load_default()
        dd.rectangle([0, 0, SZ, 30], fill=(20, 20, 20))
        dd.text((6, 6), text, fill=(255, 255, 255), font=ft)
        return np.array(img)

    vmax = np.percentile(dpk[dpk > 0], 99)
    p_act = panel(act[Z], "FDG activity (MedImages SUV)", cmap_hot=True)
    p_loc = panel(local[Z], "MedImages local-deposition", vmax=vmax, cmap_hot=True)
    p_dpk = panel(dpk[Z], "DPK reference (F-18)", vmax=vmax, cmap_hot=True)
    # signed difference
    diff = (local[Z] - dpk[Z])
    dv = np.percentile(np.abs(dpk[dpk > 0]), 99)
    dn2 = np.clip(diff / (dv + 1e-9), -1, 1)
    dn2 = np.rot90(dn2)
    rr = np.clip(dn2, 0, 1); bb = np.clip(-dn2, 0, 1)
    drgb = (np.stack([rr, np.zeros_like(dn2), bb], -1) * 255).astype(np.uint8)
    dimg = Image.fromarray(drgb).resize((SZ, SZ), Image.BILINEAR)
    dd = ImageDraw.Draw(dimg)
    try:
        ft = ImageFont.truetype(FONT, 16)
    except Exception:
        ft = ImageFont.load_default()
    dd.rectangle([0, 0, SZ, 30], fill=(20, 20, 20))
    dd.text((6, 6), f"local-DPK (r={pear:.3f})  red=over blue=under", fill=(255, 255, 255), font=ft)
    p_diff = np.array(dimg)

    sep = np.full((SZ, 4, 3), 255, np.uint8)
    grid = np.hstack([p_act, sep, p_loc, sep, p_dpk, sep, p_diff])
    out = os.path.join(OUTDIR, "Dosimetry_MedImages_vs_DPK.png")
    Image.fromarray(grid).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
