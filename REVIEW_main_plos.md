# Critical Review of MedImages.jl Manuscript

**Target Journal:** *Computer Methods and Programs in Biomedicine* (CMPB)

This review evaluates the manuscript "MedImages.jl: A Julia Package for Standardized Medical Image Handling and Spatial Metadata Management" from a critical perspective. The review is structured section-by-section, focusing on scientific accuracy, clarity, accessibility for a broad biomedical informatics audience, and strict adherence to the target journal's guidelines.

---

## 1. General Formatting & CMPB Guidelines Compliance (MAJOR ISSUES)

Before delving into the narrative, the manuscript fundamentally fails to meet the basic formatting requirements for *Computer Methods and Programs in Biomedicine*. A complete structural overhaul is required before submission.

*   **Abstract Structure:** CMPB requires a structured abstract (max 350 words) with explicit headings: *Background and Objective*, *Methods*, *Results*, *Conclusions*. The current abstract is unstructured and reads more like a promotional pitch than a scientific summary.
*   **Article Structure:** CMPB requires the following distinct sections: *Introduction*, *Methods*, *Results*, *Discussion*, *Acknowledgements* (including declarations). The current manuscript uses non-standard sections like "Clinical Relevance and Multimodality Support" and "Design and Implementation" scattered confusingly. "Design and Implementation" should be "Methods".
*   **Word Count:** The manuscript is extremely dense. CMPB recommends ~3500 words for original research. The authors need to significantly condense the philosophical arguments and focus on the technical implementation and results.
*   **Missing Declarations:** CMPB strictly requires statements for:
    *   Ethics approval (even if n/a, it must be stated why).
    *   Declaration of competing interests.
    *   Funding sources.
    *   Declaration of generative AI use.
    *   *These are entirely absent.*
*   **Title Page:** The title page lacks a designated corresponding author with full postal address and telephone number (only email is provided).
*   **Author Summary:** This section is an artifact of PLOS formatting and is *not required or appropriate* for CMPB. Delete it.
*   **Title:** Too long and jargon-heavy.
    *   *Suggestion:* "MedImages.jl: A high-performance, differentiable Julia framework for medical image processing and metadata preservation"

---

## 2. Section-by-Section Critique

### Abstract

**Critique:**
The abstract is highly aggressive, using subjective terms like "eliminates," "zero-cost," and comparing itself negatively to "hybrid Python/C++ frameworks." This tone is inappropriate for a scientific paper and sounds like an advertisement. The phrase "zero-cost metadata abstractions" is computer science jargon that will alienate clinical readers. What does a 30-138x speedup mean practically for a hospital?

**Actionable Advice & Rewrite:**
*   Remove the attack on other frameworks (SimpleITK, MONAI). Focus on what MedImages.jl *does*, not what others *don't*.
*   Structure it according to CMPB guidelines.

*Suggested Rewrite:*
> **Background and Objective:** Quantitative medical imaging, particularly in Nuclear Medicine, requires software that ensures both high-throughput computational performance and strict spatial metadata integrity. Existing frameworks often face a "two-language" bottleneck, bridging high-level APIs with compiled backends, which complicates automatic differentiation (AD) and can lead to metadata drift during preprocessing. We introduce MedImages.jl, a native Julia ecosystem for 3D/4D medical image analysis.
> **Methods:** MedImages.jl implements a typed data structure that intrinsically binds voxel data to spatial metadata (origin, spacing, direction). Core geometric transformations are implemented as native, differentiable Julia kernels compatible with GPU acceleration (KernelAbstractions.jl) and SciML AD engines (Zygote.jl). The framework integrates HDF5 for high-throughput I/O and supports batched multimodality operations.
> **Results:** In a 100-subject PET/CT preprocessing benchmark, MedImages.jl demonstrated a 7.2× total turnaround speedup compared to standard Python-based caching pipelines. GPU-accelerated spatial transformations achieved up to 135× speedups over CPU baselines. We validated end-to-end differentiability through a spatial transformer network and multi-organ affine registration, and confirmed that Standardized Uptake Value (SUV) calculations match established clinical tools to double-precision accuracy.
> **Conclusions:** MedImages.jl provides a performant, fully differentiable environment for medical image processing. By unifying spatial operations and metadata management within a single language, it facilitates large-scale clinical research and physics-informed deep learning without compromising clinical validity.

### Introduction

**Critique:**
The "Four Converging Challenges" structure is overly dramatic and reads like a manifesto. The terms "metadata preservation crisis" and "synchronization tax" are hyperbole. The introduction assumes the reader intimately understands the "two-language problem" in Julia. For a general biomedical informatics audience (CMPB), you must explain *why* this matters clinically before diving into LLVM JIT compilation.

*   *Clarity Issue:* "The inherent 'volume' of 3D modalities puts significant stress..." Volume of what? Data? Physical space? Be precise.
*   *Jargon:* "LLVM Just-In-Time (JIT) compilation", "highly fused and optimized kernels". This needs translation for clinicians. Explain that it compiles code on-the-fly to run optimally on the specific hardware, bypassing the need for pre-compiled C++ libraries.

**Actionable Advice:**
*   Condense the four challenges into a standard, cohesive narrative flow: (1) The rise of large, multi-modal datasets in NM; (2) The computational bottleneck in preprocessing; (3) The specific problem of metadata loss when moving to deep learning arrays; (4) The need for differentiability in modern AI.
*   Tone down the rhetoric. Replace "crisis" with "challenge."

### Clinical Relevance and Multimodality Support (Merge into Introduction/Background)

**Critique:**
This section feels disjointed. It's a textbook explanation of Nuclear Medicine (PET, SPECT, MRI) placed awkwardly in the middle of a software paper. The CMPB audience generally knows what PET and SPECT are.

*   *Focus:* You don't need to define 18F-FDG or PSMA. Instead, focus *only* on why these modalities stress computational pipelines (e.g., "Whole-body PET/CT generates massive 4D datasets requiring precise spatiotemporal alignment").

**Actionable Advice:**
*   Drastically reduce the generic NM tutorial. Move the relevant parts (SUV calculations, metadata necessity) into the Introduction to justify the software's design.

### Design and Implementation (Rename to Methods)

**Critique:**
This section is too brief on *how* things are actually implemented and too heavy on buzzwords ("BIDS-inspired," "zero-cost abstractions").
*   *Missing Detail:* How exactly does `BatchedMedImage` handle different spatial geometries within the same batch? If it's a 4D tensor `(x, y, z, batch)`, all items must have the same voxel dimensions. The text implies it handles heterogeneous metadata, which is confusing if it's a single tensor. This must be explicitly clarified.
*   *Code Snippets:* A software paper in CMPB heavily benefits from a short, clear code snippet showing the API (e.g., loading an image, rotating it, calculating SUV). The current manuscript has zero code.

**Actionable Advice:**
*   Rename to "Methods".
*   Add a small code block demonstrating a typical workflow.
*   Clearly explain the tensor structure of `BatchedMedImage` and how metadata is vectorized.

### Results

**Critique:**
The results section is a mixed bag. The benchmarks are impressive, but the presentation is messy.

*   **Challenge 1 (Volume):** Comparing MedImages.jl (HDF5) to MONAI (PersistentDataset) is slightly apples-to-oranges. You are comparing a data format (HDF5) to a caching mechanism (Pickle/Pt). The text claims "7.2x speedup," but is that because Julia is faster, or just because HDF5 reads faster than MONAI's default cache? You must acknowledge this nuance.
*   **Proof-of-Concept (Learned Inverse Rotation):** This is a great toy example, but it's buried in text. A flowchart or diagram showing the forward and backward pass through the `interpolate_pure` kernel is desperately needed here.
*   **Multi-Organ Affine Registration:** This is a strong experiment, but the description is incredibly dense. "Intra-workgroup reduction uses binary tree reduction in shared memory..." is way too deep in the weeds for a general biomedical audience. Focus on the *outcome* and the *fact* that it's differentiable, rather than the CUDA-level thread mechanics.
*   **Quantitative Dosimetry Refinement (SciML):** This subsection is excellent, but it introduces three complex architectures (FNO, PINN, UDE) very rapidly. The "Explainability" claims are slightly overstated—a PINN doesn't inherently "provide the highest theoretical explainability," it just enforces a constraint. Be careful with absolute claims.
*   **Figure 1 & 2 (SUV Consistency & Batched Rotation):** The text mentions "nearest-neighbor interpolation was strictly applied" for masks. Was the Z-rotation and X-translation applied to the mask and the PET/CT simultaneously? The explanation of *how* the `BatchedMedImage` achieves this under the hood is missing.

**Actionable Advice:**
*   Create a table summarizing the SciML experiments (Architecture, Loss, MSE, Advantage) to make the text less dense.
*   Clarify the HDF5 vs MONAI benchmark conditions.
*   Add a diagram for the Differentiable Registration pipeline.

### Discussion

**Critique:**
The discussion reads like a polemic against Python. While highlighting advantages is necessary, the attacks on MONAI, TorchIO, and Deepali are overly hostile. Statements like "This dependency chain maintains a layer of indirection that can lead to memory management issues" require citations or direct experimental proof in the paper, otherwise they are just speculation.

*   *Tone:* The section "Addressing Challenge 4... Beyond the Python-C++ Divide" sounds arrogant. "MedImages.jl represents a paradigm shift" is a massive claim for a new software package. Let the community decide if it's a paradigm shift.
*   *Table 2:* Calling SimpleITK's interoperability gap "Discarded upon conversion to NumPy" is unfair; SimpleITK *has* metadata, users just choose to throw it away. MedImages.jl prevents this by design, which is a good point, but phrase it as "Design enforces metadata retention vs. User-dependent retention."

**Actionable Advice:**
*   Adopt a more objective, academic tone. Focus on trade-offs. What are the *disadvantages* of Julia? (e.g., Time-to-first-plot/compilation latency, smaller ecosystem compared to PyTorch). Acknowledging limitations builds trust. A paper that claims a tool has zero downsides is suspicious.
*   Add a dedicated "Limitations" subsection. Mentioning the heavy memory requirements for testing (as noted in the project files) or the learning curve of Julia would be highly appropriate here.

### Conclusion

**Critique:**
"The ability to perform fully differentiable spatial transformations while maintaining absolute zero-cost metadata integrity is not just a technical achievement; it is a clinical necessity..." This is dramatic flair.

**Actionable Advice:**
Keep it factual and summarize the main contributions: high performance, differentiability, and metadata safety, and how these enable advanced SciML in medical imaging.

---

## Summary of Required Actions for CMPB Submission

1.  **Restructure entirely:** Abstract (Structured), Introduction, Methods, Results, Discussion, Conclusion.
2.  **Add Declarations:** Ethics, Competing Interests, Funding, GenAI.
3.  **Tone adjustment:** Remove hyperbole (crisis, zero-cost, paradigm shift) and direct attacks on other frameworks.
4.  **Add Visuals/Code:** Include a workflow diagram and a short code snippet demonstrating the API.
5.  **Add Limitations:** Discuss the compilation overhead or memory footprint of the Julia/GPU stack.
6.  **Clarify `BatchedMedImage`:** Explain the tensor mechanics of handling batched spatial metadata.