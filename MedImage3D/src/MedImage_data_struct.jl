"""
Here we define necessary data structures for the project.
Main data structure is a MedImage object which is a 3D image with some metadata.

!!!! Currently implemented as Struct but will be better to use as metadata arrays
"""
struct MedImage
    pixel_array
    direction
    spacing
    orientation
    origin
    date_of_saving::String
    patient_id::String
end




@enum Interpolator nearest_neighbour=0 linear=2 b_spline=3

