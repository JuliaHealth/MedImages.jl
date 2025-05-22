# MedImages.jl

![MedImages.jl JuliaHealth Logo](./docs/src/assets/logo.png)

The filesystem and choice of metadata is loosly based on BIDS format [1] .


This project was created to standardize data handling of 3D and 4D medical imaging, for now without support for ultrasonography as we are looking for contributor with expertise in this area . It is currently subject to change and I am open to suggestions that would improve the library in construction.
I've included below in categories what needs to be done and some basic ideas on How to approach it. I will post it for consultations with the community and then process to create test cases where Python SimpleItk (or other) methods will be treated as a reference.
1. Designing data structure - requirements (need to explicitly specify the most important and the rest will be just in an additional dictionary)
   * hold voxel data as a multidimensional array
   * keep spatial metadata - origin, orientation, spacing
   * type of the image - label/CT/MRI/PET (need to construct enum for this) - frequently will be needed to be supplied manually
   * subtype of the image for example if MRI ADC/DWI/T2 etc.  (need to construct enum for this)
   * type of the voxel data (for example Float32)
   * Date of saving
   * Acquisition time
   * Patient ID if present
   * current device - (CPU, GPU)
   * Study UID
   * Patient UID
   * Series UID
   * Study Description
   * Original file name
   * Display data - set of colors for the labels; window value for CT scan - will provide set of defaults based on image type and I will modify MedEye3D for convenient visualizations.
   * Clinical data dictionary - age, gender ...
   * Is contrast administered
   * The rest of the metadata loaded from a file to store in a dictionary 
1. Data loading
   * Nifti - start with Nifti.jl
   * Dicom - Dicom.jl
   * Mha - ?
1. Modifying voxel data together with spacing data
  * Modifying orientation of all images to single orientation - for example, RAS (we should select some default orientation) - TODO with AxisArrays.jl plus keep track of origin
  * Changing spacing to a given spacing - TODO with AxisArrays.jl with Interpolations.jl plus keep track of origin; use nearest neighbor interpolator for the label; and for example b spline for others
  * Resampling to another grid - for example, resample PET image to CT image based just on spatial metadata keeping track of changed metadata of the moving image TODO with AxisArrays.jl and Interpolations.jl
  * Cropping and dilatating with adjusting of origin offset TODO with AxisArrays.jl, storing also positions of original image in metadata (some artifacts happen on the image edges)
1. Adding persistency
   * store the data as array in HDF5 and metadata as its attributes
   * design efficient loading and saving irrespective of the device the array is on
   * add the possibility to save back as nifty and dicom files for exporting
  
Point 3 is the trickiest one to do as there are a lot of corner cases
to look for https://github.com/haberdashPI/MetaArrays.jl

## References
1)https://www.nature.com/articles/sdata201644

  
   
