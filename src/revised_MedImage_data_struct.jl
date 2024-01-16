using Pkg
Pkg.add(["Dictionaries"])
using Dictionaries

struct MedImage
end

setImageMetaDataFromNifti(object::MedImage)= begin

end


#constructor function for nifti files
MedImage(nifti_data_value_array::Array{Any}) = begin
  return MedImage(...nifti_data_value_array)
end


#constructor function for dicom files
  
