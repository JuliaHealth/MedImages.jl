"""
some issues that may occur and also why python rotation implementation is so clumsy here 
is described in https://stackoverflow.com/questions/56171643/simpleitk-rotation-of-volumetric-data-e-g-mri
we have separate python function fro z rotation and rotations in other planes

Also transformations are nicely shown on couple first slides of 
https://www.cs.cornell.edu/courses/cs4620/2010fa/lectures/03transforms3d.pdf

"""


# include("../src/MedImage_data_struct.jl")
include("../src/Load_and_save.jl")
include("./test_visualize.jl")
include("./dicom_nifti.jl")

# using .dicom_nifti

using NIfTI,LinearAlgebra,DICOM
using PythonCall


# using CondaPkg
# CondaPkg.add("simpleitk")
# CondaPkg.add("numpy")
# CondaPkg.add_pip("simpleitk", version="")

sitk = pyimport("SimpleITK")
np = pyimport("numpy")


# python implementation taken from https://stackoverflow.com/questions/56171643/simpleitk-rotation-of-volumetric-data-e-g-mri
function matrix_from_axis_angle(a)
    """ Compute rotation matrix from axis-angle.
    This is called exponential map or Rodrigues' formula.
    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])

    # This is equivalent to
    # R = (np.eye(3) * np.cos(theta) +
    #      (1.0 - np.cos(theta)) * a[:3, np.newaxis].dot(a[np.newaxis, :3]) +
    #      cross_product_matrix(a[:3]) * np.sin(theta))

    return R
end #matrix_from_axis_angle

function resample(image, transform)
    """
    This function resamples (updates) an image using a specified transform
    :param image: The sitk image we are trying to transform
    :param transform: An sitk transform (ex. resizing, rotation, etc.
    :return: The transformed sitk image
    """
    reference_image = image
    interpolator = sitk.sitkLinear
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)
end#resample

function get_center(img)
    """
    from python to test
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint(np.ceil(width/2), np.ceil(height/2), np.ceil(depth/2))
end #get_center


function rotation3d(image, axis, theta)

    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively
    :return: The rotated image
    """
    theta = np.deg2rad(theta)
    euler_transform = sitk.Euler3DTransform()
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)

    direction = image.GetDirection()

    if(axis==3)
        axis_angle = (direction[2], direction[5], direction[8], theta)
    elseif (axis==2)
        axis_angle = (direction[1], direction[4], direction[7], theta)
    elseif (axis==1)
        axis_angle = (direction[0], direction[3], direction[6], theta)
    end    
    np_rot_mat = matrix_from_axis_angle(axis_angle)
    euler_transform.SetMatrix([np_rot_mat[0][0],np_rot_mat[0][1],np_rot_mat[0][2]
                                ,np_rot_mat[1][0],np_rot_mat[1][1],np_rot_mat[1][2] 
                                ,np_rot_mat[2][0],np_rot_mat[2][1],np_rot_mat[2][2] ])
    resampled_image = resample(image, euler_transform)
    return resampled_image
end #rotation3d


# function rotate_metadata(image, axis, theta)


#     rots=[0.0,0.0,0.0]
#     rots[axis]=theta

#     # Create a rotation transform
#     dimension = image.GetDimension()
#     transform = sitk.Euler3DTransform()
#     transform.SetRotation(np.deg2rad(rots[1]), np.deg2rad(rots[2]), np.deg2rad(rots[3]))  

#     # Create the TransformGeometryImageFilter and set the transform
#     transform.SetCenter(
#         image.TransformContinuousIndexToPhysicalPoint(
#             [(sz - 1) / 2 for sz in image.GetSize()]
#         )
#     )
    
#     rotated_image = sitk.TransformGeometry(image, transform)

#     return rotated_image


# end #rotate_metadata    


function test_single_rotation(medIm,sitk_image, axis::Int, theta::Float64,debug_folder_path,dummy_run=false)
    """
    test if the rotation of the image lead to correct change in the pixel array
    and the metadata the operation will be tasted against Python simple itk function
    
    """
    #sitk implementation
    rotated=rotation3d(sitk_image,axis, theta)

    if(dummy_run)
        sitk.WriteImage(rotated, "$(debug_folder_path)/rotated_$(axis)_$(theta)_arr.nii.gz")
        return
    end
    
    #our Julia implementation
    medIm=rotate_mi([medIm],axis,theta,linear)[0]
    test_object_equality(medIm,rotated)



end #test_single_rotation   




"""
testing rotations against Python simple itk function

"""
function test_rotation(path_nifti,debug_folder_path,dummy_run=false)
    
    #we test rotations of diffrent exes and of diffrent angles
    for ax in [1,2,3]
        for theta in [30.0,60.0,90.0,180.0,270.0,360.0,400.0]
            #purposfully reloading each time to avoid issues with pixel 
            #array mutation
            
            #load image only in real run
            med_im=[]
            if(!dummy_run)
                med_im=load_image(path_nifti)
            end

            sitk_image=sitk.ReadImage(path_nifti)
            test_single_rotation(med_im,sitk_image, ax, theta,debug_folder_path,dummy_run)
        end#for    
    end#for


end


# image = sitk.ReadImage(imagePath)


################################################# cropping tests

"""
crop image using simple itk function and return the cropped image 
both beginning and size are tuples of 3 elements (x,y,z)
in case of begining it will mean first voxel and size how big will be the chunk to extract

"""

function sitk_crop(sitk_image,beginning,size)
    # extract = sitk.ExtractImageFilter()

    extracted_image=sitk.RegionOfInterest(sitk_image,[size[1],size[2],size[3]], [beginning[1],beginning[2],beginning[3]] )
    # extract.SetSize([size[1],size[2],size[3]])
    # extract.SetIndex([beginning[1],beginning[2],beginning[3]])
    # extracted_image = extract.Execute(sitk_image)

    print(extracted_image.GetSize())
    print(sitk_image.GetSize())
    return extracted_image
end#sitk_crop


function test_single_crop(medIm,sitk_image, begining, size,debug_folder_path,dummy_run)
    #sitk implementation
    
    cropped=sitk_crop(sitk_image,begining, size)
    if(dummy_run)
        sitk.WriteImage(cropped, "$(debug_folder_path)/cropped_$(begining)_$(size).nii.gz")
        return
    end#dummy_run        
    #our Julia implementation
    medIm=crop_mi([medIm],begining,size,linear)[0]

    test_object_equality(medIm,cropped)

end #test_single_rotation   


function test_crops(path_nifti,debug_folder_path=" ",dummy_run=false)    
    """
    test if the cropping of the image lead to correct change in the pixel array
    and the metadata the operation will be tasted against Python simple itk function

    """   
    for begining in [(0,0,0),(15,17,7)]
        for size in [(151,156,50),(150,150,53),(148,191,56)]
            med_im=[]
            if(!dummy_run)
                med_im=load_image(path_nifti)
            end
            sitk_image=sitk.ReadImage(path_nifti)
            test_single_crop(med_im,sitk_image, begining, size,debug_folder_path,dummy_run)
        end#for    
    end#for

end



######################### padding tests
"""
pad image using simple itk function and return the cropped image 
both beginning and end pad are tuples of 3 elements (x,y,z)
in case of begining it will mean first voxel and size how big will be the chunk to extract

"""

function sitk_pad(sitk_image,pad_beg,pad_end,pad_val)
    extract = sitk.ConstantPadImageFilter()
    extract.SetConstant(pad_val)
    extract.SetPadLowerBound(pad_beg)
    extract.SetPadUpperBound(pad_end)     
    extracted_image = extract.Execute(sitk_image)
    return extracted_image
end#sitk_crop

function test_pads(path_nifti,debug_folder_path,dummy_run=false)    
    """
    test if the padding of the image lead to correct change in the pixel array
    and the metadata the operation will be tasted against Python simple itk function

    """   
    for pad_beg in [(10,11,13),(15,17,19)]
        for pad_end in [(10,11,13),(15,17,19),(30,31,32)]
            for pad_val in [0.0,111.5]
                sitk_image=sitk.ReadImage(path_nifti)
                sitk_padded=sitk_pad(sitk_image, pad_beg, pad_end,pad_val)
                if(dummy_run)
                    sitk.WriteImage(sitk_padded, "$(debug_folder_path)/padded_$(pad_beg)_$(pad_end).nii.gz")
                else
                    medIm=load_image(path_nifti)
                    mi_padded=pad_mi(medIm,pad_beg,pad_end,pad_val,linear)
                    test_object_equality(mi_padded,sitk_padded)
                end

            end#for    
        end#for    
    end#for

end




####################### translation tests


"""
reference sitk translate function
"""
function sitk_translate(image,translate_by,translate_in_axis)
    translatee=[0,0,0]
    translatee[translate_in_axis]=translate_by
    transform=sitk.TranslationTransform(3,translatee )
    # reference_image = image 
    # extracted_image=sitk.Resample(image, reference_image, transform,
    # sitk.sitkLinear, 0.0)
    res=sitk.TransformGeometry(image, transform)
    
    return res
end#sitk_translate




"""
test if the translation of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function

"""
function test_translate(path_nifti,debug_folder_path,dummy_run)
    
    # Load the image from path

    for t_val in [1,10,16]
        for axis in [1,2,3]
            sitk_image=sitk.ReadImage(path_nifti)
            sitk_trnanslated=sitk_translate(sitk_image,t_val, axis)
            if(dummy_run)
                sitk.WriteImage(sitk_image, "$(debug_folder_path)/translated_$(t_val)_$(axis).nii.gz")
            else
                medIm=load_image(path_nifti)
                medIm=translate_mi(medIm,t_val, axis,linear)
                test_object_equality(medIm,sitk_trnanslated)
            end    
        end#for    
    end#for

    

end





################################################# scaling tests





"""
test if the scaling of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function

"""

# Scale the image using SimpleITK
function sitk_scale(image, zoom)
    scale_transform = sitk.ScaleTransform(3, [zoom, zoom, zoom])

    res=sitk.Resample(image, scale_transform, sitk.sitkBSpline, 0.0)

    return res
end

# Test if the scaling of the image leads to correct changes in the pixel array and metadata
function test_scale(path_nifti, debug_folder_path,dummy_run)
    for zoom in [0.6,0.9,1.0,1.3,1.8]
        sitk_image = sitk.ReadImage(path_nifti)
        sitk_scaled = sitk_scale(sitk_image, zoom)
        if(dummy_run)
            sitk.WriteImage(sitk_scaled, "$(debug_folder_path)/scaled_$(zoom).nii.gz")
        else
            medIm = load_image(path_nifti)
            medIm = scale_mi(medIm, zoom,linear)
            test_object_equality(medIm, sitk_scaled)
        end
        

    end #for 

end

# imagePath = "/workspaces/MedImage.jl/test_data/volume-0.nii.gz"
# debug_folder_path = "/workspaces/MedImage.jl/test_data/debug"
# test_scale(imagePath,  debug_folder_path,true)
  


