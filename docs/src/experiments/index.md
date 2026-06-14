# MedImages.jl Experiments

This section details the experiments conducted to evaluate the MedImages.jl framework against four core challenges in modern quantitative medical imaging. These challenges are detailed in our PLOS Computational Biology manuscript.

## Overview of Challenges

1.  **Challenge 1: Volume -- Scaling to 100 Cases (HDF5 vs. Caching)**
    Processing large biobank-scale datasets creates significant I/O and transform bottlenecks. We benchmarked a 100-case multimodal preprocessing pipeline against established caching mechanisms.
    *   [Read the Volume Scaling Experiment Details](volume_scaling.md)
2.  **Challenge 2: Speed -- GPU Acceleration Benchmarks**
    Execution speed of basic spatial filters limits rapid prototyping. We evaluated native Julia GPU acceleration against classical C++ (SimpleITK) CPU baselines for resampling and affine transformations.
    *   [Read the GPU Acceleration Experiment Details](gpu_acceleration.md)
3.  **Challenge 3: Differentiability and SciML Validation**
    Emerging Scientific Machine Learning (SciML) paradigms require fully differentiable pipelines. We validated this through two complex setups:
    *   [Learned Inverse Rotation (Geometric CNN)](learned_inverse_rotation.md): Training a network to predict and invert 3D rotations by differentiating through the spatial resampling grid.
    *   [Quantitative Dosimetry via Universal Differential Equations (UDE)](dosimetry_ude.md): A hybrid physics-neural network model to predict high-fidelity radiation dose maps, integrating spatial gradients and tissue densities.
4.  **Challenge 4: Metadata Management and Clinical Fidelity**
    To avoid "metadata drift" during preprocessing, spatial and temporal clinical context must remain bound to the voxel array. We evaluated how our framework strictly preserves these attributes.
    *   [Read the Metadata Fidelity Experiment Details](metadata_fidelity.md)
