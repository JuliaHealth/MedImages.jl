"""
given two MedImage objects and a Interpolator enum value return the moving MedImage object resampled to the fixed MedImage object
images should have the same orientation origin and spacing; their pixel arrays should have the same shape
"""
function resample_to_image(im_fixed::Array{MedImage},im_moving::Array{MedImage} ,Interpolator::Interpolator )::Array{MedImage}



end#scale_mi    