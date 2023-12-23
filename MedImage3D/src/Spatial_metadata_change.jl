"""
functions to change the metadata of a 3D image like change the orientation of the image
change spaciing to desired etc 
"""

"""
given a MedImage object and desired spacing (spacing) return the MedImage object with the new spacing

"""
function resample_to_spacing(im::Array{MedImage}, new_spacing::Tuple{Float64, Float64, Float64} )::Array{MedImage}

end#resample_to_spacing

"""
given a MedImage object and desired orientation encoded as 3 letter string (like RAS or LPS) return the MedImage object with the new oeientation
"""
function change_orientation(im::Array{MedImage}, new_orientation::String)::Array{MedImage}

end#change_orientation



