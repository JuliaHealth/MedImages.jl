include("./MedImage_data_struct.jl")
include("./Utils.jl")
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
    points_to_interpolate = get_base_indicies_arr(new_size)

    points_to_interpolate=points_to_interpolate.-1
    points_to_interpolate=points_to_interpolate.*new_spacing
    points_to_interpolate=points_to_interpolate.+1

    # print("\n ppppp $(points_to_interpolate) \n ")

    interpolated_points=interpolate_my(points_to_interpolate,im.voxel_data,old_spacing,interpolator_enum,true)

    new_voxel_data=reshape(interpolated_points,(new_size[1],new_size[2],new_size[3]))
    # new_spacing=(new_spacing[3],new_spacing[2],new_spacing[1])
    # Create the new MedImage object
    new_im =update_voxel_and_spatial_data(im, new_voxel_data
    ,im.origin,new_spacing,im.direction)

    return new_im
end#resample_to_spacing














#  te force solution get all direction combinetions put it in sitk and try all possible ways to permute and reverse axis to get the same result as sitk then save the result in json or sth and use; do the same with the origin
#     Additionally in similar manner save all directions in a form of a vector and associate it with 3 letter codes


"""
given a MedImage object and desired orientation encoded as 3 letter string (like RAS or LPS) return the MedImage object with the new orientation
"""
function change_orientation(im::MedImage, new_orientation::Orientation_code)::MedImage
    orientation_dict = Dict(
        "RAS" => [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        "LPS" => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "LAS" => [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        "RSP" => [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "LAI" => [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
        "RAI" => [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]
    )

    if !haskey(orientation_dict, new_orientation)
        error("Invalid orientation: $new_orientation")
    end

    new_direction = orientation_dict[new_orientation]

    # Determine the axes to reverse and permute based on the new direction
    reverse_axes = findall(x -> x < 0, new_direction)
    permute_order = sortperm(abs.(new_direction))

    # Apply the reverse and permute operations to the voxel data
    new_voxel_data = im.voxel_data
    for axis in reverse_axes
        new_voxel_data = reverse(new_voxel_data, dims=axis)
    end
    new_voxel_data = permutedims(new_voxel_data, permute_order)

    # Update the origin to be consistent with the changed direction
    new_origin = im.origin[permute_order]
    for axis in reverse_axes
        new_origin[axis] = im.origin[axis] + (size(im.voxel_data, axis) - 1) * im.spacing[axis]
    end
    # Create a new MedImage with the new direction and the same voxel data, origin, and spacing
    new_im = update_voxel_and_spatial_data(im, new_voxel_data
    ,new_origin,im.spacing,new_direction)

    return new_im
  end#change_orientation





# im_fixed=load_image("/home/jakubmitura/projects/MedImage.jl/test_data/volume-0.nii.gz")
# imm_res=resample_to_spacing(im_fixed, (1.0,2.0,3.0),Linear_en)

# # sitk = pyimport_conda("SimpleITK","simpleITK")
# # # function create_nii_from_medimage(med_image::MedImage, file_path::String)
# # #     # Convert voxel_data to a numpy array (Assuming voxel_data is stored in Julia array format)
# # #     # voxel_data_np = np.array(med_image.voxel_data)
    
# # #     # Create a SimpleITK image from numpy array
# # #     image_sitk = sitk.GetImageFromArray(med_image.voxel_data)
    
# # #     # Set spatial metadata
# # #     image_sitk.SetOrigin(med_image.origin)
# # #     image_sitk.SetSpacing(med_image.spacing)
# # #     image_sitk.SetDirection(med_image.direction)
    
# # #     # Save the image as .nii.gz
# # #     sitk.WriteImage(image_sitk, file_path)
# # # end


# voxel_arr=permutedims(imm_res.voxel_data,(3,2,1))
# image_sitk = sitk.GetImageFromArray(voxel_arr)

# image_sitk.SetOrigin(imm_res.origin)
# image_sitk.SetSpacing(imm_res.spacing)
# image_sitk.SetDirection(imm_res.direction)
# sitk.WriteImage(image_sitk, "/home/jakubmitura/projects/MedImage.jl/test_data/debug/resampled_sitk.nii.gz")
# create_nii_from_medimage(imm_res,"/home/jakubmitura/projects/MedImage.jl/test_data/debug/resampled_medimage.nii.gz")

# size(imm_res.voxel_data)
# # range(1, stop=5, length=100,step=0.1)