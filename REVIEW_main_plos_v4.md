# Round 2: Harsh Scientific Critique (Internal Audit)

**Manuscript:** MedImages.jl (v4)
**Focus:** Mathematical Rigor, Baseline Integrity, and Data Consistency

---

## 1. The "Baselines of Convenience" Trap
*   **Critique:** You compare your UDE ($r=0.957$) against several Python baselines (SemiDose, Spect0, DblurDoseNet) where some achieved near-zero or negative correlations. 
*   **Harsh Point:** A critical reviewer will ask: "Why did you include baselines that clearly failed to train?" (e.g., SemiDose $r=-0.69$). Including a "failed" baseline as a competitor is often seen as a way to inflate the perceived performance of the main model.
*   **Action:** You must either fix the training of these baselines or explicitly state in the Methods why these specific architectures are ill-suited for $^{177}$Lu-PSMA (e.g., inability to handle the high dynamic range of Activity vs. Density without the UDE's mechanistic prior). Don't just list them; explain the failure mode.

## 2. Mathematical Disconnect in UDE Scaling
*   **Critique:** The manuscript mentions $64^3$ patches but doesn't explain how the blood compartment ($A_{blood}$) is handled. 
*   **Harsh Point:** $A_{blood}$ is a global state, but you train on patches. How is the total uptake $\sum \text{Uptake}$ calculated during patch-wise training? If you are only summing uptake within the patch, the blood kinetics are physically incorrect during the forward pass.
*   **Action:** Clarify if $A_{blood}$ is a fixed drive function (from global PK) or if you are using a "representative volume" scaling factor. This is a major physical integrity hole.

## 3. "Zero-Cost" vs. "JIT Overhead"
*   **Critique:** You emphasize "zero-cost abstractions" in the Intro but admit to "time-to-first-plot" JIT latency in the Limitations.
*   **Harsh Point:** You can't have it both ways. The "cost" of the abstraction is paid at compile-time. For a clinical user, a 30-second JIT pause is not "zero-cost." 
*   **Action:** Refine the terminology. Use **"Zero-Cost Runtime Abstraction"** to specify that once compiled, the performance is optimal. Acknowledge that Julia's benefit is for **high-throughput loops**, not single-image GUI interactions.

## 4. Figure 2: The "Log Scale" Deception
*   **Critique:** Your speed comparison plot (Fig 2) uses a log scale.
*   **Harsh Point:** While scientifically valid, reviewers often view log scales in performance plots as a way to hide smaller variances or exaggerate differences. 
*   **Action:** Add a linear-scale inset or clearly state the absolute delta (e.g., 80ms difference) in the caption. For a single patch, is 80ms actually "clinically significant"? You must argue that for **Ensemble Uncertainty Quantification** (10,000 runs), this 10x difference is the difference between minutes and hours.

## 5. Statistical Power ($N=48$)
*   **Critique:** Validation $N=48$ patches. 
*   **Harsh Point:** From how many patients? If these 48 patches come from only 2-3 patients, they are spatially correlated and the $p < 0.001$ is misleading.
*   **Action:** Specify the number of independent patients in the validation set. If it's low, acknowledge this as a "Proof of Concept" study rather than a "Large-scale Validation."

---

**Status:** **Minor Revision Required** to close physical logic gaps.
