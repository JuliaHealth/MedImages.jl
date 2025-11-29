module Spatial_metadata_change
using Interpolations

using ..MedImage_data_struct, ..Utils, ..Orientation_dicts, ..Load_and_save
export change_orientation, resample_to_spacing
"""
given a MedImage object and desired spacing (spacing) return the MedImage object with the new spacing
"""
function scale(itp::AbstractInterpolation{T,N,IT}, ranges::Vararg{AbstractRange,N}) where {T,N,IT}
    # overwriting this function becouse check_ranges giving error
    # check_ranges(itpflag(itp), axes(itp), ranges)
    ScaledInterpolation{T,N,typeof(itp),IT,typeof(ranges)}(itp, ranges)
end




function resample_to_spacing(im, new_spacing::Tuple{Float64,Float64,Float64}, interpolator_enum, use_cuda=false)::MedImage
    old_spacing = im.spacing
    old_spacing = (old_spacing[3], old_spacing[2], old_spacing[1])
    new_spacing = (new_spacing[3], new_spacing[2], new_spacing[1])
    old_size = size(im.voxel_data)
    new_size = Tuple{Int,Int,Int}(ceil.((old_size .* old_spacing) ./ new_spacing))

    # Use optimized kernel resampling
    new_voxel_data = resample_kernel_launch(im.voxel_data, old_spacing, new_spacing, new_size, interpolator_enum)

    new_spacing = (new_spacing[3], new_spacing[2], new_spacing[1])
    new_im = Load_and_save.update_voxel_and_spatial_data(im, new_voxel_data, im.origin, new_spacing, im.direction)

    return new_im
end#resample_to_spacing




#  te force solution get all direction combinetions put it in sitk and try all possible ways to permute and reverse axis to get the same result as sitk then save the result in json or sth and use; do the same with the origin
#     Additionally in similar manner save all directions in a form of a vector and associate it with 3 letter codes


"""
given a MedImage object and desired orientation encoded as 3 letter string (like RAS or LPS) return the MedImage object with the new orientation
"""
function change_orientation(im::MedImage, new_orientation::Orientation_code)::MedImage
    old_orientation = Orientation_dicts.number_to_enum_orientation_dict[im.direction]
    reorient_operation = Orientation_dicts.orientation_pair_to_operation_dict[(old_orientation, new_orientation)]
    return change_orientation_main(im, new_orientation, reorient_operation)
end#change_orientation



function change_orientation_main(im::MedImage, new_orientation::Orientation_code, reorient_operation)::MedImage
    perm = reorient_operation[1]
    reverse_axes = reorient_operation[2]
    origin_transforms = reorient_operation[3]
    spacing_transforms = reorient_operation[4]

    origin1 = copy(collect(im.origin))

    sizz = size(im.voxel_data)
    spacing1 = copy(collect(im.spacing))

    res_origin = [0.0, 0.0, 0.0]

    for origin_axis in [1, 2, 3]

        spac = collect(spacing1)
        spac_axis, sizz_axis, prim_origin_axis, op_sign = origin_transforms[origin_axis]
        res_origin[origin_axis] = origin1[prim_origin_axis] + ((spac[spac_axis] * (sizz[sizz_axis] - 1)) * op_sign)


    end


    # first we need to permute the voxel data to match the desired orientation
    im_voxel_data = copy(im.voxel_data)
    if (length(perm) > 0)
        im_voxel_data = permutedims(im_voxel_data, (perm[1], perm[2], perm[3]))
    end

    if (length(reverse_axes) == 1)
        im_voxel_data = reverse(im_voxel_data; dims=reverse_axes[1])
    elseif (length(reverse_axes) > 1)
        im_voxel_data = reverse(im_voxel_data; dims=Tuple(reverse_axes))
    end






    # now we need to change spacing as needed
    st = spacing_transforms
    sp = im.spacing
    new_spacing = (sp[st[1]], sp[st[2]], sp[st[3]])
    new_im = Load_and_save.update_voxel_and_spatial_data(im, im_voxel_data, res_origin, new_spacing, orientation_dict_enum_to_number[new_orientation])

    # print("\n res_origin $(res_origin) \n")
    return new_im
end#change_orientation





# im_fixed=load_image("/home/jakubmitura/projects/MedImage.jl/test_data/volume-0.nii.gz")
# imm_res=resample_to_spacing(im_fixed, (1.0,2.0,3.0),Linear_en)

# # sitk = pyimport("SimpleITK")
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
end#Spatial_metadata_change



# using KernelAbstractions, Test
# @kernel function mul2_kernel(A)
#     I = @index(Global)
#     shared_arr=@localmem(Float32, (512,3))
#     index_local = @index(Local, Linear)
#     A[index_local] = 2 * A[I]
#   end

#   dev = CPU()
# A = ones(1024, 1024)
# ev = mul2_kernel(dev, 64)(A, ndrange=size(A))
# synchronize(dev)
# all(A .== 2.0)
