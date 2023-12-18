"""
module implementing basic transformations on 3D images 
like translation, rotation, scaling, and cropping both input and output are MedImage objects
"""

"""
given a MedImage object and a Tuple that contains the rotation values for each axis (x,y,z in order)
return the rotated MedImage object 
"""
function rotate_mi(im::MedImage, rotate_by::Tuple{Float64, Float64, Float64})::MedImage



end#rotate_mi    



"""
given a MedImage object and a Tuples that contains the the number of voxels we want to remove in each axs (x,y,z in order)
from the begining of each axis (crop_beg) and from the end of each axis (crop_end)
return the cropped MedImage object 
"""
function crop_mi(im::MedImage, crop_beg::Tuple{Int64, Int64, Int64},crop_end::Tuple{Int64, Int64, Int64}  )::MedImage



end#crop_mi    


"""
given a MedImage object translation value and axis in witch to translate the image return translated image
"""
function crop_mi(im::MedImage, crop_beg::Tuple{Int64, Int64, Int64},crop_end::Tuple{Int64, Int64, Int64}  )::MedImage



end#crop_mi    



