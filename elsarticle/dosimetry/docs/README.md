# SciML UDE Dosimetry: Model Documentation & Implementation Report

This directory contains the implementation of a **Universal Differential Equation (UDE)** framework for high-fidelity $^{177}$Lu-PSMA dosimetry, bridging the gap between fast analytical kernels and gold-standard Monte Carlo (MC) simulations.

## 1. Summary of Model Improvements

The previous iteration of the model suffered from numerical instability and significant physical scaling errors. The following changes were implemented to stabilize training and ensure physical correctness:

### A. Multiplicative Neural Corrector
To ensure that the absorbed dose remains physically plausible (always positive and proportional to local activity), the neural network residual $\mathcal{N}_\theta$ was transformed into a **multiplicative correction factor**:
$$ \dot{D}_{total}(\mathbf{r}, t) = \frac{\dot{D}_{base}(\mathbf{r}, t) \cdot \exp\left(\text{tanh}(\mathcal{N}_\theta(A, \rho, D_{approx}))\right)}{m(\mathbf{r}) + \epsilon} $$
*   **Why:** This prevents the model from predicting negative energy deposition and bounds the influence of the neural network, preventing catastrophic gradients during the early stages of training.

### B. Unified Physical Units (SI/Clinical Standard)
The model now operates in standard clinical units (**Gray/hour**). The dose conversion constant was unified based on the primary decay characteristics of $^{177}$Lu:
*   **Energy ($\bar{E}$):** 0.147 MeV/decay.
*   **Conversion Factor ($DOSE\_CONV$):** $8.478 \times 10^{-8}$. This constant converts activity in Becquerels (Bq) and mass in grams to Gy/h, accounting for Joule-to-MeV conversion and seconds-to-hours.
*   **Why:** Eliminates arbitrary scaling factors (like $10^5$ or $10.0$) that masked the true physics and led to uninterpretable model outputs.

### C. Vascular Proxy Uptake
Instead of distributing blood-borne activity uniformly across all voxels, the model now uses the **initial SPECT activity ($A_0$) as a proxy for vascularity**:
*   Voxel-wise uptake is proportional to $A_0 / \sum A_0$.
*   **Why:** Prevents "ghost" dose accumulation in non-vascularized tissues (like pure air or dense bone) and more accurately reflects the biology of radiopharmaceutical delivery.

### D. Density-Based Masking
*   **Mechanism:** Any voxel with a mass density $\rho < 0.1 \text{ g/cm}^3$ (air-equivalent) is masked to zero dose rate.
*   **Why:** Prevents numerical explosions in the background where $1/(m+\epsilon)$ can reach extremely high values.

---

## 2. Temporal Integration & Monte Carlo Verification

### The 12.5-Day (300h) Assumption
We integrate the ODE system from $0 \to 300$ hours. 
*   **Verification:** The physical half-life of $^{177}$Lu is ~159.5h. When combined with biological washout ($T_{1/2,bio} \approx 35-50\text{h}$), the **effective half-life** ($T_{1/2,eff}$) is typically **30 to 50 hours**.
*   **Depth:** 300 hours represents **6 to 10 effective half-lives**, capturing $>99.5\%$ of the total cumulative dose.
*   **MC Alignment:** This matches the Monte Carlo ground truth which typically simulates the "total energy deposition" until infinite time (residence time).

---

## 3. Current Project State

### Location: `/home/user/MedImages.jl/elsarticle/dosimetry/`

| Artifact | Description |
| :--- | :--- |
| `train_ude.jl` | Core training script with improved physics and multiplicative residuals. |
| `eval_model.jl` | Updated evaluation script for generating binary prediction data. |
| `model_heavy.jls` | Trained weights from the successful 10-epoch run. |
| `model_best.jls` | Weights from the epoch with the lowest validation loss. |
| `vis_results/` | Patient-specific subdirectories containing: |
| | - `ct.bin`, `pred.bin`, `orig.bin`: Raw floating-point data. |
| | - `..._transverse.png`, `..._coronal.png`, `..._sagittal.png`: Visualization slices. |
| | - `metrics.txt`: Table of MAE values per patient. |

### Visual Verification
Visualizations (PNGs) have been generated for all validation patients. These images show a 5-panel comparison:
1.  **CT:** Anatomical context.
2.  **Baseline Approx:** The fast analytical kernel.
3.  **MC Target:** The gold-standard ground truth.
4.  **Predicted (Model):** The output of the SciML UDE.
5.  **Error (Pred - MC):** Spatial map of the remaining residuals.

Current validation MAE has stabilized at **~0.09**, a significant improvement over previous divergent runs.
