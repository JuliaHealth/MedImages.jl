#!/usr/bin/env python3
"""fig15 — why MedImages and SimpleITK looked like they disagreed on rotation.

Row 1 (one consistent pipeline, real CT via SimpleITK): a +45 deg Euler transform
spins the CONTENT clockwise, a -45 deg transform spins it counter-clockwise --
because SimpleITK's Resample maps output->input. MedImages rotate_mi(+45) uses the
standard +theta = counter-clockwise convention, so it equals SimpleITK's -45.

Row 2: the actual committed MedImages render (original -> rotate_mi(+45)), plus the
validated pixel-perfect result: Pearson(MedImages+45, SimpleITK-45) = 1.0000 (fig3).
"""
import math, numpy as np, SimpleITK as sitk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO = "/home/dorachan/code/juliahealth/MedImages.jl-mitura"
OUT  = f"{REPO}/paper_figures/fig15_rotation_sign_convention.png"
CT   = f"{REPO}/test_data/volume-0.nii.gz"
MED_ORIG = f"{REPO}/test/visual_output/screenshots/CT_1_original.png"
MED_ROT  = f"{REPO}/test/visual_output/screenshots/CT_2_rotated_45deg.png"
AIR = -1000.0

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans", "savefig.bbox": "tight"})


def ctwin(a, lo=-160, hi=240):
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)


def rot(img, deg):
    c = img.GetSize()
    t = sitk.Euler3DTransform()
    t.SetCenter(img.TransformIndexToPhysicalPoint([c[0]//2, c[1]//2, c[2]//2]))
    t.SetRotation(0, 0, math.radians(deg))
    return sitk.GetArrayFromImage(sitk.Resample(img, img, t, sitk.sitkLinear, AIR))


def main():
    img = sitk.ReadImage(CT)
    a0  = sitk.GetArrayFromImage(img)
    Z   = int(np.argmax([np.sum(a0[z] > -200) for z in range(a0.shape[0])]))
    p45 = rot(img, +45.0)
    m45 = rot(img, -45.0)
    # confirm the two SimpleITK signs are genuinely different, and describe direction
    r_pm = np.corrcoef(ctwin(a0[Z]).ravel(), ctwin(p45[Z]).ravel())[0, 1]

    fig, ax = plt.subplots(2, 3, figsize=(14.5, 10), constrained_layout=True)

    # ---- Row 1: one pipeline, opposite signs ----
    row1 = [
        (a0[Z],  "Original CT", None),
        (p45[Z], "SimpleITK  SetRotation(+45°)", "content spins  CLOCKWISE ↻"),
        (m45[Z], "SimpleITK  SetRotation(−45°)", "content spins  COUNTER-CLOCKWISE ↺"),
    ]
    for a, (im, title, sub) in zip(ax[0], row1):
        a.imshow(np.rot90(ctwin(im)), cmap="gray", interpolation="bilinear", aspect="equal")
        a.set_title(title, fontsize=12, fontweight="bold")
        if sub:
            a.set_xlabel(sub, fontsize=11,
                         color=("#b00020" if "CLOCK" in sub and "COUNTER" not in sub else "#0a7d00"))
        a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("SAME pipeline\n(real CT, SimpleITK)", fontsize=12, fontweight="bold")

    # ---- Row 2: real MedImages output + the verdict ----
    ax[1, 0].imshow(mpimg.imread(MED_ORIG), aspect="equal")
    ax[1, 0].set_title("MedImages — original\n(actual render)", fontsize=12, fontweight="bold")
    ax[1, 1].imshow(mpimg.imread(MED_ROT), aspect="equal")
    ax[1, 1].set_title("MedImages  rotate_mi(axis=3, +45°)\n(actual render — CCW)",
                       fontsize=12, fontweight="bold", color="#0a7d00")
    for a in (ax[1, 0], ax[1, 1]):
        a.set_xticks([]); a.set_yticks([])

    ax[1, 2].axis("off")
    ax[1, 2].text(0.02, 0.5,
                  "The sign convention\n"
                  "─────────────────\n"
                  "SimpleITK's Resample maps\n"
                  "output→input, so a +θ transform\n"
                  "rotates the IMAGE by −θ.\n\n"
                  "MedImages rotate_mi uses the\n"
                  "standard  +θ = counter-clockwise.\n\n"
                  "⇒  MedImages(+45°)  ≡  SimpleITK(−45°)\n\n"
                  "Verified pixel-perfect on the real CT:\n"
                  "   Pearson = 1.0000   (see fig3)\n\n"
                  "The earlier 'they disagree' was purely\n"
                  "this ± transform-sign mismatch.",
                  fontsize=11.5, va="center", ha="left", family="DejaVu Sans")

    fig.suptitle("Rotation sign convention  —  MedImages rotate_mi(+45°) ≡ SimpleITK SetRotation(−45°), pixel-perfect (Pearson = 1.0000)",
                 fontsize=14.5, fontweight="bold")
    fig.savefig(OUT)
    print(f"SimpleITK +45 vs original Pearson = {r_pm:.3f} (low = genuinely rotated)")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
