```@raw html
---
layout: home

hero:
  name: "MedImages.jl"
  text: "Medical Image Processing in Julia"
  tagline: A comprehensive Julia package for loading, transforming, and analyzing 3D/4D medical images with proper spatial metadata handling
  image:
    src: /logo.png
    alt: MedImages.jl Logo
  actions:
    - theme: brand
      text: Get Started
      link: /manual/get_started
    - theme: alt
      text: API Reference
      link: /api
    - theme: alt
      text: View on GitHub
      link: https://github.com/JuliaHealth/MedImages.jl

features:
  - icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"></path><circle cx="12" cy="13" r="3"></circle></svg>
    title: Complete I/O Support
    details: Load and save NIfTI files with full spatial metadata preservation. HDF5 storage for efficient dataset management.
  - icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
    title: Spatial Transformations
    details: Rotation, cropping, padding, scaling, and translation with automatic origin/spacing adjustment.
  - icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"></path><path d="M2 17l10 5 10-5"></path><path d="M2 12l10 5 10-5"></path></svg>
    title: Resampling & Registration
    details: Resample to new spacing or register images to a common coordinate space with multiple interpolation methods.
  - icon: <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"></path><path d="M21 3v5h-5"></path></svg>
    title: Orientation Management
    details: Convert between 8 standard orientations (LPS, RAS, etc.) with proper coordinate system handling.
---
```

````@raw html
<div class="vp-doc" style="width:80%; margin:auto">

<p style="margin-bottom:2cm"></p>

<h1>What is MedImages.jl?</h1>

<p>
MedImages.jl is a Julia package for standardizing the handling of 3D and 4D medical images. It provides a unified data structure that combines voxel data with comprehensive spatial and clinical metadata, ensuring that transformations maintain physical accuracy.
</p>

<h2>Key Features</h2>

<ul>
<li><strong>Unified Data Structure:</strong> The <code>MedImage</code> struct encapsulates voxel data, spatial metadata (origin, spacing, direction), and clinical metadata in a single container.</li>
<li><strong>Coordinate System Aware:</strong> All operations maintain proper relationships between array indices and physical coordinates.</li>
<li><strong>Multiple Interpolation Methods:</strong> Nearest neighbor, linear, and B-spline interpolation for different use cases.</li>
<li><strong>Orientation Support:</strong> Work with 8 standard orientations and convert between them seamlessly.</li>
<li><strong>ITK/SimpleITK Compatible:</strong> Uses ITKIOWrapper for I/O, ensuring compatibility with the broader medical imaging ecosystem.</li>
</ul>

<h2>Quick Start</h2>

<h3>Installation</h3>

```julia
using Pkg
Pkg.add("MedImages")
```

<h3>Basic Usage</h3>

```julia
using MedImages

# Load a CT scan
ct = load_image("/path/to/scan.nii.gz", "CT")

# Inspect properties
println("Dimensions: ", size(ct.voxel_data))
println("Spacing (mm): ", ct.spacing)
println("Origin (mm): ", ct.origin)

# Resample to isotropic spacing
isotropic = resample_to_spacing(ct, (1.0, 1.0, 1.0), Linear_en)

# Save result
create_nii_from_medimage(isotropic, "/output/isotropic_ct")
```

<h2>Supported Modalities</h2>

<table>
<tr><th>Modality</th><th>Type</th><th>Subtypes</th></tr>
<tr><td>CT</td><td><code>CT_type</code></td><td>CT_subtype</td></tr>
<tr><td>MRI</td><td><code>MRI_type</code></td><td>T1, T2, FLAIR, ADC, DWI</td></tr>
<tr><td>PET</td><td><code>PET_type</code></td><td>FDG, PSMA</td></tr>
</table>

<h2>Available Transformations</h2>

<table>
<tr><th>Function</th><th>Description</th></tr>
<tr><td><code>crop_mi</code></td><td>Extract a region of interest</td></tr>
<tr><td><code>pad_mi</code></td><td>Add voxels around the image</td></tr>
<tr><td><code>rotate_mi</code></td><td>Rotate around an axis</td></tr>
<tr><td><code>translate_mi</code></td><td>Shift the image origin</td></tr>
<tr><td><code>scale_mi</code></td><td>Resize the image</td></tr>
<tr><td><code>resample_to_spacing</code></td><td>Change voxel spacing</td></tr>
<tr><td><code>resample_to_image</code></td><td>Register to another image</td></tr>
<tr><td><code>change_orientation</code></td><td>Convert between orientations</td></tr>
</table>

<h2>Integration</h2>

<p>MedImages.jl is designed to work with:</p>

<ul>
<li><a href="https://github.com/JuliaHealth/ITKIOWrapper.jl">ITKIOWrapper.jl</a> - For DICOM and NIfTI I/O</li>
<li><a href="https://github.com/JuliaHealth/MedEye3d.jl">MedEye3d.jl</a> - For 3D visualization</li>
<li><a href="https://github.com/JuliaHealth/MedPipe3D.jl">MedPipe3D.jl</a> - For processing pipelines</li>
</ul>

<h2>Documentation Sections</h2>

<ul>
<li><a href="/manual/get_started">Getting Started</a> - Installation and setup instructions</li>
<li><a href="/manual/tutorials">Tutorials</a> - Step-by-step guides for common workflows</li>
<li><a href="/manual/code_example">Code Examples</a> - Ready-to-use code snippets</li>
<li><a href="/api">API Reference</a> - Complete function documentation</li>
<li><a href="/reference/data_structures">Data Structures</a> - Type and enum documentation</li>
</ul>

<h2>Contributing</h2>

<p>
Contributions are welcome! Please visit our <a href="https://github.com/JuliaHealth/MedImages.jl">GitHub repository</a> to:
</p>

<ul>
<li>Report issues</li>
<li>Submit pull requests</li>
<li>Suggest new features</li>
</ul>

<h2>License</h2>

<p>MedImages.jl is released under the MIT License.</p>

<div style="text-align: center; margin-top: 4rem; padding: 2rem 0; border-top: 1px solid #eaecef; color: #4e6e8e;">
Part of the <a href="https://juliahealth.org">JuliaHealth</a> organization
</div>

</div>
````
