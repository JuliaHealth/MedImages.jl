# MedImages.jl Infographic Design Logic

This document provides a precise, step-by-step blueprint for visualizing the four core challenges addressed by `MedImages.jl` in the `old_plos.tex` manuscript, as well as a dedicated section for the Quantitative UDE Dosimetry Experiment. Each section strictly follows a four-part narrative flow.

---

## 1. Challenge 1: The Volume Bottleneck (Biobank-Scale Processing)

### Core Narrative Flow
1. **Challenge (Issue):** High-throughput preprocessing of biobank-scale multimodal datasets (thousands of 3D/4D studies) is a major hardware and I/O bottleneck in modern clinical research.
2. **Knowledge Gap (Other Solutions):** Traditional caching solutions in Python, such as MONAI's `PersistentDataset` or TorchIO, rely on heavy Pickle/Pt dict-based serialization. This creates immense memory buildup and I/O friction, severely bottlenecking large-scale pipelines and preventing true biobank-scale deployment.
3. **How Addressed:** `MedImages.jl` solves this by implementing zero-serialization HDF5 persistence combined with Fused Affine GPU kernels.
4. **How Experiments Prove the Point:** Our 100-subject benchmark experiment proved a **7.2× faster turnaround time** (~90 ms vs ~650 ms per subject), enabling true biobank-scale data ingestion in minutes rather than days without cache penalties.

### Pictographic Representation & Layout
1. **Challenge Panel (The Input Domain):**
    *   *Visuals:* A representation of a massive biobank.
    *   *Layout:* Top-level label reading "Biobank dataset: 10,000+ multimodal studies".
2. **Knowledge Gap Panel (Left - Traditional Pipeline):**
    *   *Iconography:* A slow-moving funnel or a series of stacked, disconnected disk drives representing `Pickle/Pt Caching`. Draw a "memory leak" or "bottleneck" warning sign attached to highlight TorchIO's memory buildup issues and MONAI's Python caching limits.
    *   *Text:* "MONAI PersistentDataset (~650 ms)"
    *   *Connections:* A dashed, red, and thick arrow (representing I/O friction) points downwards.
3. **How Addressed Panel (Right - MedImages Pipeline):**
    *   *Iconography:* A sleek, single solid state drive (HDF5) connected directly to a GPU microchip icon. Attach a "Zero-Serialization" badge.
    *   *Text:* "MedImages.jl HDF5 + Native GPU (~90 ms)"
    *   *Connections:* A solid, green, and direct arrow (representing massive I/O throughput) points downwards.
4. **How Experiments Prove the Point Panel (The Results Node):**
    *   *Visuals:* Both the red and green arrows converge on a large, central target node below them.
    *   *Text:* Bold text reading "**7.2× Speedup, Unlocking Thousands of Studies**".
    *   *Clinical Integration (Real Scanner Data):* Replace the central "Results Node" vault icon with an actual Maximum Intensity Projection (MIP) rendering of a whole-body [177Lu]Lu-PSMA SPECT/CT study sourced directly from the clinical biobank, visually anchoring the scale of the dataset to real patient anatomy.

Try to get the image as similar as possible to /home/user/MedImages.jl/elsarticle/figures_new/examples/ch1.png

---

## 2. Challenge 2: The Two-Language Barrier (Execution Speed)

### Core Narrative Flow
1. **Challenge (Issue):** The classic "Two-Language Problem" forces medical researchers to prototype in high-level languages while relying on opaque, pre-compiled low-level binaries for performance.
2. **Knowledge Gap (Other Solutions):** Python ecosystems rely heavily on wrapping C++ libraries (e.g., SimpleITK). These opaque binaries act as a "brick wall," fundamentally limiting native GPU acceleration, creating CPU bottlenecks for spatial operations, and preventing deep framework inspection or custom hardware compilation.
3. **How Addressed:** `MedImages.jl` is written in pure Julia, compiled directly via LLVM JIT. This unified approach inherently unlocks native hardware acceleration via `KernelAbstractions.jl`.
4. **How Experiments Prove the Point:** Our experiments on target hardware (NVIDIA RTX 3090, Intel Core Ultra 9) demonstrate massive performance gains: an **135× speedup** for Fused Affine transformations (0.83 ms vs 6.69 ms on CPU) and a **115× speedup** for spatial resampling compared to standard Python/C++ baselines.

### Pictographic Representation & Layout
1. **Challenge Panel (The Input Problem):**
    *   *Visuals:* The classic "Two-Language Problem" visualized as a vertical split layout separating high-level prototyping from low-level execution.
2. **Knowledge Gap Panel (Top Region - Python Ecosystem):**
    *   *Iconography:* Two distinctly colored interlocking puzzle pieces. One piece labeled "Python (High-Level)", the other labeled "C++ (SimpleITK)".
    *   *Connection:* A literal "brick wall" graphic or a thick black vertical line between the Python script icon and a GPU icon, symbolizing the barrier to hardware acceleration.
    *   *Text:* Place a "SimpleITK (CPU)" box next to the brick wall.
3. **How Addressed Panel (Bottom Region - Julia Ecosystem):**
    *   *Iconography:* A single, unified, glowing crystal or glowing gear labeled "Pure Julia / LLVM JIT".
    *   *Connection:* Direct, continuous gradient arrows flowing from the unified gear into a GPU chip icon.
    *   *Text:* Place a "MedImages GPU" box.
4. **How Experiments Prove the Point Panel (Performance Comparison):**
    *   *Visuals:* Place a speed dial/speedometer icon next to the GPU. The needle is pinned to the maximum redline.
    *   *Text:* "**135× Fused Affine Acceleration**" and "**115× Resampling Acceleration**".
    *   *Clinical Integration (Real Scanner Data):* Instead of standard progress bars to show the 6.69 ms vs 0.83 ms difference, use a sequence of real CT cross-sections undergoing rapid spatial resampling (e.g., scaling from $512\times512$ to $128\times128$). The CPU side shows a single frame rendering slowly, while the MedImages GPU side displays a rapid cascade of successfully transformed clinical overlays.

try to make it as similar as possible to /home/user/MedImages.jl/elsarticle/figures_new/examples/ch2.png
---

## 3. Challenge 3: Differentiability (Physics-in-the-Loop UDEs)

### Core Narrative Flow
1. **Challenge (Issue):** Accurately modeling physical phenomena (like quantitative voxel-level dosimetry) requires integrating known scientific equations directly into the training loop of machine learning architectures.
2. **Knowledge Gap (Other Solutions):** Pure deep learning ("black box" CNNs/U-Nets) fails to respect physical constraints. Conversely, traditional analytical clinical models (like VSV) assume homogeneous environments and ignore critical tissue heterogeneity. Furthermore, Python's fragmented "Walled Gardens" (PyTorch/JAX) struggle to differentiate through arbitrary mechanistic simulators.
3. **How Addressed:** We implement a 4-State Universal Differential Equation (UDE) in Julia, natively integrating Mechanistic Knowns with Learned Uncertainties (Neural Residuals) using `Zygote.jl`.
4. **How Experiments Prove the Point:** Our experiments proved this architecture connects Julia's ecosystem (`DifferentialEquations.jl`, `Lux.jl`, `MedImages.jl`) to achieve Monte Carlo-level accuracy (**Pearson $r=0.957$**) while avoiding the "Walled Garden" problem.

### Pictographic Representation & Layout
1. **Challenge Panel (The Objective):**
    *   *Visuals:* The goal of integrating knowns and unknowns into a single quantitative pipeline.
2. **Knowledge Gap Panel (The Walled Garden & Limitations):**
    *   *Iconography:* Place icons for PyTorch and JAX inside a locked cage ("Walled Garden") connected by broken arrows to illustrate standard framework limitations (inability to differentiate through mechanistic simulators).
    *   *Text:* "Black Boxes" fail physical constraints; Analytical models ignore heterogeneity.
3. **How Addressed Panel (The Inputs & UDE Integrator):**
    *   *Left Input Node (The Knowns):* A rigid, solid blue square containing mathematical symbols ($S_{homo}$, $\lambda_{phys}$, $CF$, $\rho$). Labeled "Mechanistic Physics".
    *   *Right Input Node (The Unknowns):* A flexible, dashed orange cloud or brain icon containing a neural network perceptron diagram ($\mathcal{N}_\theta$). Labeled "Neural Residual Corrector".
    *   *Central Hub (The UDE Integrator):* A solid blue arrow from the Left Node and a dashed orange arrow from the Right Node converge into a large, spinning gear surrounding a stylized integral symbol ($\int$). Show continuous circular arrows representing "Multiple Dispatch / Expression Problem Solved." Labeled "Julia UDE Integrator (SciML)".
4. **How Experiments Prove the Point Panel (The Output Node):**
    *   *Visuals:* An arrow flows outward from the Central Hub to a final target node.
    *   *Text:* A green checkmark badge reading "**Pearson r = 0.957 (Monte Carlo Fidelity)**".
    *   *Clinical Integration (Real Scanner Data):* The final target node should be an authentic, pseudo-colored dose map overlay on an axial CT slice, generated directly from the scanner's DICOM data via the UDE pipeline, explicitly demonstrating real-world tissue heterogeneity.

Try to make it as simillar as possible to /home/user/MedImages.jl/elsarticle/figures_new/examples/ch3.png
---

## 4. Challenge 4: Metadata Management (Theranostic Batched Processing)

### Core Narrative Flow
1. **Challenge (Issue):** Complex theranostic workflows require perfectly aligning highly heterogeneous, multi-modal spatial data (e.g., mapping SPECT AC, SPECT NAC, and Dosemaps to a single CT anatomical grid).
2. **Knowledge Gap (Other Solutions):** In standard Python pipelines, spatial metadata (origin, spacing, direction) is easily decoupled and lost the moment a medical image (e.g., SimpleITK) is converted into a raw NumPy tensor for deep learning. This "Metadata Drift" leads to catastrophic spatial misalignment and quantitative errors downstream.
3. **How Addressed:** `MedImages.jl` solves this via the `BatchedMedImage` structure, which explicitly binds physical metadata to the 4D voxel tensor using Julia's rigorous type system.
4. **How Experiments Prove the Point:** Our compound affine transformation experiments demonstrated flawless quantitative alignment, ensuring Standardized Uptake Value (SUV) consistency with **< 1.5% deviation** across massive multimodal batches.

### Pictographic Representation & Layout
1. **Challenge Panel (The Complex Workflow):**
    *   *Visuals:* A representation of heterogeneous, multi-modal spatial data that must be aligned.
2. **Knowledge Gap Panel (Metadata Drift):**
    *   *Visuals:* Show a `sitk.GetArrayFromImage()` operation acting like a pair of scissors, slicing off a "Spacing/Origin" tag and dropping it into a trash bin (NumPy Array conversion) to contrast standard Python methods resulting in data misalignment.
3. **How Addressed Panel (The Protected BIDS-Inspired Tensor):**
    *   *Top Bounding Box:* A large, transparent 3D cube containing a vertical stack of four distinct 2D slice images ("CT Anatomy", "177Lu Dosemap", "177Lu NAC", "177Lu AC").
    *   *Metadata Coupling:* Draw rigid, solid brackets tying the entire stack of slices together. Attached to the bracket, place a "Data Tag" icon locked inside a shield (Julia's Type System) explicitly listing the protected physical properties: "Origin (x,y,z)", "Spacing (x,y,z)", "Direction Matrix".
    *   *The Transform:* Draw a thick, downward-pointing arrow from the Top Box, wrapped in a circular rotation vector (symbolizing compound 45° rotation and grid resampling) pointing to the bottom results stack.
4. **How Experiments Prove the Point Panel (The Aligned Output):**
    *   *Visuals:* At the bottom, redraw the same stack of 4 slices, but rotated 45 degrees. Place a "magnifying glass" over the bottom stack showing a graph of Standardized Uptake Values (SUV).
    *   *Text Badge:* "Clinical Metadata Perfectly Synchronized: SUV Consistency < 1.5% Deviation".
    *   *Clinical Integration (Real Scanner Data):* The 2D slice stacks (both the original and the rotated 45-degree version) must use genuine, coregistered clinical data. Specifically, show real clinical NIfTI slices corresponding to the patient's CT Anatomy, 177Lu-PSMA Dosemap, and SPECT (AC/NAC) arrays, visualizing the exact coordinate matrices being protected.

try to make it as similar as possible to /home/user/MedImages.jl/elsarticle/figures_new/examples/ch4.png
---

## 5. Quantitative UDE Dosimetry Experiment

This section strictly details the experimental methodology and superior performance of the 4-State UDE model for 177Lu-PSMA dosimetry.

### Core Narrative Flow
1. **Challenge (Issue):** The clinical objective is to accurately map functional SPECT and anatomical CT data to high-fidelity Monte Carlo (MC) ground truth dosimetry.
2. **Knowledge Gap (Other Solutions):**
    * *Clinical Baseline:* Voxel S-Value (VSV) convolution over Time-Integrated Activity (TIA) using Python (`Evaluate-Proj` / `pytheranostics`) ignores tissue heterogeneity.
    * *Deep Learning Baseline:* Pure 3D CNN / U-Net architectures ("Black Boxes" like DblurDoseNet) struggle with physical constraints.
3. **How Addressed:** The SciML/Julia "No-Approx UDE" model cleanly isolates known physics (primary scatter) from a neural residual.
4. **How Experiments Prove the Point:** Experiments show the clinical VSV baseline achieves a Pearson correlation of $r=0.912$, and pure deep learning achieves $r \approx 0.557$. In stark contrast, the SciML UDE achieves state-of-the-art **$r=0.957$** while maintaining a **10× speed advantage** over traditional Python analytical frameworks.

### Pictographic Representation & Layout
1. **Challenge Panel (The Objective Definition):**
    *   *Header:* "High-Fidelity 177Lu-PSMA Dosimetry Comparison"
    *   *Inputs:* Shared raw SPECT/CT inputs representing functional and anatomical patient data.
2. **Knowledge Gap Panel (The Competitors):**
    *   *Lane 1 (Left - Pure Deep Learning):* A black box labeled "3D U-Net / DblurDoseNet". Raw SPECT/CT data points in. A non-linear, poorly constrained arrow points out to a low-fidelity dose map exhibiting widespread unconstrained spatial artifacts.
    *   *Lane 2 (Center - Clinical Analytical / Python):* A calculator or standard rigid gears labeled "VSV Convolution (PyTheranostics)". SPECT TIA maps point in. Straight arrow points out to a homogeneous dose map lacking high-frequency anatomical definition at tissue boundaries.
3. **How Addressed Panel (Lane 3 - SciML UDE / Julia):**
    *   *Visuals:* A hybrid icon combining a math equation ($S_{homo}$) and a neural network ($\mathcal{N}_\theta$) encased in a golden shield. SPECT, CT (HU $\to \rho$), and physical constants point in. A glowing, thick green arrow points out to a highly detailed, precise dose map (matching the Monte Carlo Ground Truth).
4. **How Experiments Prove the Point Panel (Metrics and Clinical Overlays):**
    *   *Metrics Badges:*
        *   Lane 1: Red color, "Pearson r = 0.557 (Fails physical constraints)".
        *   Lane 2: Yellow color, "Pearson r = 0.912 (Ignores tissue heterogeneity)".
        *   Lane 3: Green color, "**State-of-the-Art: Pearson r = 0.957**".
    *   *Speed Metric:* A horizontal bar or speedometer spanning the bottom connecting Lane 2 (Python VSV) to Lane 3 (Julia UDE). Text: "MedImages.jl / SciML architecture maintains a **10× Speed Advantage** over traditional Python analytical frameworks."
    *   *Clinical Integration (Real Scanner Data):* For the three map outputs in their respective lanes, embed real 2D axial dose profiles (isodose contours overlaid on anatomical CT) derived directly from the $177$Lu-PSMA patient cohort. This starkly contrasts the unconstrained neural artifacts of Lane 1, the homogeneous lack of boundary detail in Lane 2, and the high-fidelity anatomical constraints achieved in Lane 3 against Monte Carlo ground truth.

try to make it as similar as possible to /home/user/MedImages.jl/elsarticle/figures_new/examples/ch5.png
---


Use icons from 
https://lucide.dev/
https://tabler.io/icons
https://heroicons.com/
