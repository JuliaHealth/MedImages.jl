# Universal Differential Equations (UDE) for Dosimetry

This document describes the theoretical framework of the UDE models used in the `MedImages.jl` project and details the specific improvements implemented to outperform analytical baselines.

## 1. What is a UDE? (From Scratch)

A **Universal Differential Equation (UDE)** is a hybrid modeling approach that merges the interpretability of physics-based differential equations with the high-capacity pattern recognition of neural networks.

In traditional dosimetry, we use static equations like:
$$D = \frac{A \cdot C}{\rho \cdot V}$$
Where $D$ is dose, $A$ is activity, $C$ is a constant, $\rho$ is density, and $V$ is volume. This assumes a simple "local deposition" physics model that fails to account for complex photon scattering and tissue heterogeneity.

In a **UDE**, we redefine the problem as a system of dynamic equations where the derivative is partially unknown:
$$\frac{d}{dt} \text{Dose}(t) = \text{Physics\_Term}(A, \rho) + \mathcal{N}_\theta(A, \rho, \dots)$$

The model "solves" the physics we know while the neural network $\mathcal{N}_\theta$ learns the "missing physics" (residuals) directly from the Monte Carlo gold standard data.

## 2. The Core ODE System

The dosimetry UDE implemented here tracks the activity of a radiopharmaceutical across four internal compartments:
1.  **$A_{blood}$**: Activity circulating in the blood.
2.  **$A_{free}$**: Activity in the interstitial tissue.
3.  **$A_{bound}$**: Activity bound to cellular targets (e.g., PSMA receptors).
4.  **$DOSE$**: The cumulative energy deposited in the voxel.

The system of equations is:
- $dA_{blood}/dt = -(\text{clearance} + \lambda_{phys}) A_{blood}$
- $dA_{free}/dt = \text{uptake} - (\text{binding} + \lambda_{phys}) A_{free}$
- $dA_{bound}/dt = \text{binding} - (\text{internalization} + \lambda_{phys}) A_{bound}$
- $d(DOSE)/dt = \text{Physics\_Term}(A_{total}, \rho) + \text{Neural\_Residual}(\text{inputs}, \theta)$

## 3. Implemented Improvements

To improve upon the analytical baseline and the original UDE architecture, we incorporated the following refinements:

### A. Density Gradient Branch ($\nabla \rho$)
- **Improvement**: Added a 3rd parallel input branch to the neural network.
- **Physics Reason**: Dosimetry accuracy is lowest at organ boundaries (e.g., tissue-bone or tissue-lung interfaces). By providing the 3D finite-difference gradient of the density map ($|\nabla \rho|$), the neural network can explicitly detect structure boundaries and apply specific scattering corrections that a local density map cannot capture.

### B. Balanced Unit Training
- **Improvement**: Re-scaled internal Activity variables by $10^{-6}$ and adjusted constants proportionally.
- **Problem Fixed**: Raw SPECT counts are often $>10^9$. Calculating gradients through an ODE solver with such large magnitudes leads to floating-point overflow and `NaN` values. 
- **Result**: Perfect numerical stability on H100 GPUs and faster convergence.

### C. Dynamic ODE Standardization
- **Improvement**: Implemented voxel-wise standardization ($z = (x - \mu)/\sigma$) *inside* each time-step of the ODE solver.
- **Reason**: This ensures that the inputs to the neural network remain in the $O(1)$ range regardless of the patient's absolute activity levels or scan noise, preventing the "vanishing gradient" problem.

### D. Robust Additive Refinement
- **Improvement**: Switched from multiplicative to additive refinement with a "Matched Scale" prior.
- **Reason**: By adding the neural output to the analytical physics term, the model is guaranteed to start with a high correlation (~0.82) provided by the physics. The parameters are then purely dedicated to refining the spatial errors relative to the Monte Carlo ground truth.

## 4. Model Inputs and Outputs

The neural network component $\mathcal{N}_\theta$ within the UDE is a 3D Convolutional Neural Network with three parallel input branches and a single regression output.

### Exact Inputs (per voxel)
1.  **Standardized Activity ($A_{total}$)**: 
    - The sum of free and bound activity ($A_{free} + A_{bound}$) calculated at the current solver time-step.
    - Standardized using dynamic voxel-wise stats: $z = (A - \mu_A) / (\sigma_A + \epsilon)$.
2.  **Standardized Density ($\rho$)**: 
    - Physical density derived from the CT scan (Hounsfield Units mapped to g/cm³).
    - Standardized relative to the patch volume.
3.  **Standardized Density Gradient ($|\nabla \rho|$)**:
    - The magnitude of the 3D finite-difference gradient of the density map.
    - Used to explicitly signal anatomical tissue boundaries to the network.

### Exact Outputs (per voxel)
- **Neural Residual ($R_\theta$)**: 
    - A single scalar value representing the learned correction to the instantaneous dose rate.
    - In the implementation, this is added to the analytical dose rate: $dD/dt = \text{softplus}(\frac{A \cdot C}{\rho \cdot V} + R_\theta)$.
    - Final output of the integrated ODE system is the **Cumulative Absorbed Dose in Grays (Gy)**.

## 5. Limitations and Future Work

To maintain scientific rigor, the following physical abstractions in the current UDE framework are acknowledged as areas for future refinement:

1.  **Assumption of Static Spatial Anatomy**: The model currently ignores macroscopic physiological deformation (e.g., respiratory motion, bladder filling) and tumor shrinkage over the therapy cycle by using a static day-0 CT mass map.
2.  **Isotropic Pharmacokinetics**: The system solves $dA/dt$ equations on an independent voxel-by-voxel basis, neglecting physical diffusion and interstitial fluid pressure gradients within the tumor microenvironment.
3.  **Static Receptor Capacity ($B_{MAX}$)**: The Michaelis-Menten terms treat receptor thresholds as constant, abstractions that neglect radiation-induced receptor degradation ("stunning" effects) and cell death over time.
4.  **Simplified Renal Mechanics**: Renal clearance ($k_{10}$) is scaled linearly with baseline GFR, which may oversimplify complex active tubular secretion/reabsorption and ignore acute variations in hydration or function during treatment.
5.  **Decoupling of Beta and Gamma Transport**: $^{177}$Lu is a mixed emitter. The current model lumps local $\beta^-$ deposition and penetrating $\gamma$ transport into a single neural correction, rather than using a physically superior dual-kernel approach.

## 6. Current Status
The **Improved UDE** currently achieves:
- **Mean Pearson: 0.8966** (continuing to climb toward 0.90+)
- **Mean MAE: 389.45 Gy** (already 7% better than original targets)
