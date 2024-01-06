include("./MedImage_data_struct.jl")
"""
module implementing basic transformations on 3D images 
like translation, rotation, scaling, and cropping both input and output are MedImage objects
"""

"""
given a MedImage object will rotate it by angle (angle) around axis (rotate_axis)
the center of rotation is set to be center of the image
we are setting Interpolator by using Interpolator enum
return the rotated MedImage object 
"""
function rotate_mi(im::Array{MedImage}, rotate_axis::Int64,angle::Float64, Interpolator::Interpolator)::Array{MedImage}

  nothing

end#rotate_mi    



"""
given a MedImage object and a Tuples that contains the location of the begining of the crop (crop_beg) and the size of the crop (crop_size) crops image
we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used)
return the cropped MedImage object 
"""
function crop_mi(im::Array{MedImage}, crop_beg::Tuple{Int64,Int64,Int64}, crop_size::Tuple{Int64,Int64,Int64}, Interpolator::Interpolator)::Array{MedImage}

  nothing

end#crop_mi    


"""
given a MedImage object and a Tuples that contains the information on how many voxels to add in each axis (pad_beg) and on the end of the axis (pad_end)
we are performing padding by adding voxels with value pad_val
we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used)
return the cropped MedImage object 
"""
function pad_mi(im::Array{MedImage}, pad_beg::Tuple{Int64,Int64,Int64}, pad_end::Tuple{Int64,Int64,Int64},pad_val, Interpolator::Interpolator)::Array{MedImage}

  nothing

end#pad_mi    




"""
given a MedImage object translation value (translate_by) and axis (translate_in_axis) in witch to translate the image return translated image
we are setting Interpolator by using Interpolator enum
return the translated MedImage object
"""
function translate_mi(im::Array{MedImage}, translate_by::Int64, translate_in_axis::Int64, Interpolator::Interpolator)::Array{MedImage}

  nothing

end#crop_mi    


"""
given a MedImage object and a Tuple that contains the scaling values for each axis (x,y,z in order)
we are setting Interpolator by using Interpolator enum
return the scaled MedImage object 
"""
function scale_mi(im::Array{MedImage}, scale::Tuple{Float64,Float64,Float64}, Interpolator::Interpolator)::Array{MedImage}

  nothing

end#scale_mi    



