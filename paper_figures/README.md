# Paper Figures — MedImages.jl (Mitura fork)

Production-ready figures (300 DPI / multi-thousand-pixel) for the manuscript.
All figures reproduced on Julia 1.11 from `jakubMitura14/MedImages.jl`.

## Freshly reproduced from the fork (high-DPI, this run)

| File | Shows | Source |
|---|---|---|
| `fig1_transforms_synthetic.png` | 3×3 grid of batched synthetic transforms (rotate/translate/crop/pad/shear/scale/resample) | `test/generate_visual_samples.jl` → `experiments/make_paper_figures.py` |
| `fig2_transforms_realct.png` | Real CT: original / 45° rotation / 0.5× scale / 2 mm resample | `experiments/sciml_dose_refinement/generate_ct_screenshots.jl` → `make_paper_figures.py` |
| `fig3_rotation_vs_simpleitk.png` | **Rotation correctness vs SimpleITK — pixel-perfect, Pearson = 1.0000** (MedImages +45° CCW; SimpleITK's Resample maps output→input so its +θ transform = −θ image) | `make_paper_figures.py` |
| `fig14_medimages_vs_simpleitk_allops.png` | **MedImages vs SimpleITK across ALL operations on a real CT** — rotate / scale / resample / crop / pad, each as MedImages \| SimpleITK \| \|diff\|. Pearson: rotate 1.0000, scale 1.0000, resample 0.9939, crop 1.0000, pad 1.0000 | `cmp_ct_transforms.jl` + `make_comparison_grid.py` |
| `fig4_dosimetry_vs_dpk.png` | **MedImages dose vs F-18 dose-point-kernel** on real TCIA FDG PET/CT (body-masked Pearson 0.97; local over-estimates peaks ~5.6%) | `medimages_dose.jl` + `build_dpk_reference.py` → `make_paper_figures.py` |

## Cleaned challenge infographics (matplotlib, overlaps fixed)

The previously committed `challenge_*.png` were rendered by the **PIL** generator and had
images dropped over title text. These were regenerated with `src/render_infographics_plt.py`
(matplotlib) after removing the floating thumbnail overlays and re-spacing colliding labels.

| File | Challenge |
|---|---|
| `fig5_challenge1_volume.png` | Biobank volume / I-O bottleneck (7.2× speedup) |
| `fig6_challenge2_speed.png` | Two-language barrier / GPU acceleration |
| `fig7_challenge3_differentiability.png` | Physics-in-the-loop UDEs (r = 0.957) |
| `fig8_challenge4_metadata.png` | Metadata fidelity under compound transforms (<1.5% SUV drift) |
| `fig9_dosimetry_methods_comparison.png` | DL vs analytical vs SciML-UDE dosimetry lanes |

## Committed article charts / residual maps (reused, high-res)

Per project decision, these are the fork's committed `article/figures/` (cross-framework
benchmarks need PyTorch/JAX, and the dosimetry residuals use the protected Lu-177 cohort, so
they are not re-run here).

| File | Shows |
|---|---|
| `fig10_transform_benchmarks.png` | Per-operation transform timing (MedImages vs SimpleITK) |
| `fig11_cross_language_speed.png` | UDE forward-pass latency: DifferentialEquations.jl vs torchdiffeq vs Diffrax |
| `fig12_dosimetry_residuals_64.png` | 64³ dose residuals: MC GT vs UDE / CNN / analytical |
| `fig13_dosimetry_residuals_fullbody.png` | Full-body dose residual comparison |

## Regenerate

```bash
# scientific figures (figs 1–4)
julia +1.11 --startup-file=no --project=. test/generate_visual_samples.jl
julia +1.11 --startup-file=no --project=. experiments/sciml_dose_refinement/generate_ct_screenshots.jl
julia +1.11 --startup-file=no --project=. experiments/sciml_dose_refinement/medimages_dose.jl
python3 experiments/sciml_dose_refinement/build_dpk_reference.py \
    test_data/tcia_pet/activity.nii.gz test_data/tcia_pet/density.nii.gz test_data/tcia_pet/dose_dpk.nii.gz
python3 experiments/make_paper_figures.py

# cleaned infographics (figs 5–9)
python3 src/render_infographics_plt.py
```
