include("./MedImage_data_struct.jl")
include("./Load_and_save.jl")

using Interpolations

"""
functions to change the metadata of a 3D image like change the orientation of the image
change spaciing to desired etc 
"""

"""
given a MedImage object and desired spacing (spacing) return the MedImage object with the new spacing

"""

function scale(itp::AbstractInterpolation{T,N,IT}, ranges::Vararg{AbstractRange,N}) where {T,N,IT}
    # overwriting this function becouse check_ranges giving error
    # check_ranges(itpflag(itp), axes(itp), ranges)
    ScaledInterpolation{T,N,typeof(itp),IT,typeof(ranges)}(itp, ranges)
end

function resample_to_spacing(im::MedImage
                            ,new_spacing::Tuple{Float64,Float64,Float64}
                            ,interpolator_enum::Interpolator_enum)::MedImage
    old_spacing = im.spacing
    old_size = size(im.voxel_data)
    new_size = Tuple{Int,Int,Int}(ceil.((old_size .* old_spacing) ./ new_spacing))

    if interpolator_enum == Nearest_neighbour_en
    itp = interpolate(im.voxel_data, BSpline(Constant()))
    elseif interpolator_enum == Linear_en
        itp = interpolate(im.voxel_data, BSpline(Linear()))
    elseif interpolator_enum == B_spline_en
        itp = interpolate(im.voxel_data, BSpline(Cubic(Line(OnGrid()))))
    end
    #we indicate on each axis the spacing from area we are samplingA
    A_x1 = 1:old_spacing[1]:(old_size[1]+old_spacing[1]*old_size[1])
    A_x2 = 1:old_spacing[2]:(old_size[2]+old_spacing[2]*old_size[2])
    A_x3 = 1:old_spacing[3]:(old_size[3]+old_spacing[3]*old_size[3])
    itp = scale(itp, A_x1, A_x2,A_x3)
    # Create the new voxel data
    new_voxel_data = Array{eltype(im.voxel_data)}(undef, new_size)

    for i in 1:(new_size[1]), j in 1:(new_size[2]), k in 1:(new_size[3])


        i_1= max((((i-1) * new_spacing[1])+1),1)
        j_1= max((((j-1) * new_spacing[2])+1),1)
        k_1= max((((k-1) * new_spacing[3])+1),1)
        if((i==1))
            i_1=1#i*new_spacing[1]
        end        
        if((j==1))
            j_1=1#j*new_spacing[1]
        end        
        if((k==1))
            k_1=1
        end                
        
        new_voxel_data[i,j,k] = itp(i_1, j_1,k_1)
    end
    # new_spacing=(new_spacing[3],new_spacing[2],new_spacing[1])
    # Create the new MedImage object
    new_im =update_voxel_and_spatial_data(im, new_voxel_data
    ,im.origin,new_spacing,im.direction)

    return new_im
end#resample_to_spacing

"""
given a MedImage object and desired orientation encoded as 3 letter string (like RAS or LPS) return the MedImage object with the new orientation
"""
function change_orientation(im::MedImage, new_orientation::String)::MedImage
    # Create a dictionary to map orientation strings to direction cosines
    orientation_dict = Dict(
        "RAS" => [1 0 0; 0 1 0; 0 0 1],
        "LPS" => [-1 0 0; 0 -1 0; 0 0 -1],
        "LAS" => [-1 0 0; 0 1 0; 0 0 1],
        "RSP" => [1 0 0; 0 -1 0; 0 0 -1],
        "LAI" => [-1 0 0; 0 0 -1; 0 1 0],
        "RAI" => [1 0 0; 0 0 -1; 0 1 0]
    )

    # Check if the new orientation is valid
    if !haskey(orientation_dict, new_orientation)
        error("Invalid orientation: $new_orientation")
    end

    # Get the direction cosines for the new orientation
    new_direction = orientation_dict[new_orientation]

    # Create a new MedImage with the new direction and the same voxel data, origin, and spacing
    new_im = update_voxel_and_spatial_data(im, im.voxel_data
    ,im.origin,im.spacing,new_direction)

    return new_im
  end#change_orientation

# im_fixed=load_image("/home/jakubmitura/projects/MedImage.jl/test_data/volume-0.nii.gz")
# imm_res=resample_to_spacing(im_fixed, (1.0,2.0,3.0),Linear_en)

# sitk = pyimport_conda("SimpleITK","simpleITK")
# # function create_nii_from_medimage(med_image::MedImage, file_path::String)
# #     # Convert voxel_data to a numpy array (Assuming voxel_data is stored in Julia array format)
# #     # voxel_data_np = np.array(med_image.voxel_data)
    
# #     # Create a SimpleITK image from numpy array
# #     image_sitk = sitk.GetImageFromArray(med_image.voxel_data)
    
# #     # Set spatial metadata
# #     image_sitk.SetOrigin(med_image.origin)
# #     image_sitk.SetSpacing(med_image.spacing)
# #     image_sitk.SetDirection(med_image.direction)
    
# #     # Save the image as .nii.gz
# #     sitk.WriteImage(image_sitk, file_path)
# # end


# voxel_arr=permutedims(imm_res.voxel_data,(3,2,1))
# image_sitk = sitk.GetImageFromArray(voxel_arr)

# image_sitk.SetOrigin(imm_res.origin)
# image_sitk.SetSpacing(imm_res.spacing)
# image_sitk.SetDirection(imm_res.direction)
# sitk.WriteImage(image_sitk, "/home/jakubmitura/projects/MedImage.jl/test_data/debug/out.resampled.nii.gz")
# # create_nii_from_medimage(imm_res,"/home/jakubmitura/projects/MedImage.jl/test_data/debug/out.resampled.nii.gz")

# size(imm_res.voxel_data)
# # range(1, stop=5, length=100,step=0.1)