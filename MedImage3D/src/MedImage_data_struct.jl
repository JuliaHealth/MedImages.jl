using Pkg
Pkg.add(["Dictionaries"])
using Dictionaries







"""
Here we define necessary data structures for the project.
Main data structure is a MedImage object which is a 3D image with some metadata.

!!!! Currently implemented as Struct but will be better to use as metadata arrays
"""

#following struct can be expanded with all the relevant meta data mentioned within the readme.md of MedImage.jl
struct MedImage
    pixel_array
    spacing
    origin
    date_of_saving::String
    patient_id::String
end


#constructor function for MedImage
function MedImage(MedImage_struct_attributes::Dictionaries.Dictionary{String,Any})
    return MedImage(
        get(MedImage_struct_attributes,"pixel_array",[]),
        get(MedImage_struct_attributes,"direction",[]),
        get(MedImage_struct_attributes,"spacing",[]),
        get(MedImage_struct_attributes,"orientation",[]),
        get(MedImage_struct_attributes,"origin",[]),
        get(MedImage_struct_attributes,"date_of_saving",""),
        get(MedImage_struct_attributes,"patient_id","")      
    )
end

@enum Interpolator nearest_neighbour=0 linear=2 b_spline=3

