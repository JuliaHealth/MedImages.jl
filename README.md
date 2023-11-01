# MedImage3D.jl

This project is created to standardice data handling of 3D and 4D medical imaging. It is currently subject to change and I am open to suggestions that would improve the library in construction.
Below in categories what needs to be done and some basic ideas How to approach it. I will post it for the consultations with the community and then process to create test cases where Python SimpleItk (or other) methods will be treated as a reference.
1. Designing data structure - requirements
   * hold voxel data as multidimensional array
   * keep spatial metadata - origin, direction, spacing
   * type of the image - label/CT/MRI/PET (need to construct enum for this)
   * subtype of the image for example if MRI ADC/DWI/T2 etc.  (need to construct enum for this)
   * type of the voxel data (for example Float32)
   * current device - (CPU, GPU)
   * original file name
   * display data - set of colors for the labels; window value for CT scan 
   * the rest of the metadata in a dictionary 
1. Data loading
   * Nifti - start with Nifti.jl
   * Dicom - Dicom.jl
   * Mha - ?
1. Data loading
  
   
