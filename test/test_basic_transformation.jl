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
    return img.TransformIndexToPhysicalPoint((Py(pyconvert(Int,np.ceil(width/2))), Py(pyconvert(Int,np.ceil(height/2)))
    , Py(pyconvert(Int,np.ceil(depth/2)))))
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



function test_single_rotation(medIm::MedImage,sitk_image, axis::Int, theta::Float64,dummy_run=false)
    """
    test if the rotation of the image lead to correct change in the pixel array
    and the metadata the operation will be tasted against Python simple itk function
    
    """
    #sitk implementation
    rotated=rotation3d(sitk_image,axis, theta)
    
    if(dummy_run)
        return
    end
    #our Julia implementation
    medIm=rotate_mi([medIm],axis,theta,linear)[0]

    test_object_equality(medIm,rotated)

end #test_single_rotation   

"""
testing rotations against Python simple itk function

"""
function test_rotation(path_nifti,dummy_run=false)
    
    #we test rotations of diffrent exes and of diffrent angles
    mode= pixel_array_mode
    for ax in [1,2,3]
        for theta in [30,60,90,180,270,360,400]
            #purposfully reloading each time to avoid issues with pixel 
            #array mutation
            
            #load image only in real run
            med_im=ifelse(dummy_run,load_image(path_nifti),[])

            sitk_image=sitk.ReadImage(path_nifti)
            test_single_rotation(med_im,sitk_image, ax, theta,dummy_run)
        end#for    
    end#for
    mode = Mode_mi.metadata


end

imagePath="/workspaces/MedImage.jl/test_data/volume-0.nii.gz"
test_rotation(imagePath,true)
# image = sitk.ReadImage(imagePath)


################################################# cropping tests

"""
crop image using simple itk function and return the cropped image 
both beginning and size are tuples of 3 elements (x,y,z)
in case of begining it will mean first voxel and size how big will be the chunk to extract

"""

function sitk_crop(sitk_image,beginning,size)
    extract = sitk.ExtractImageFilter()
    extract.SetSize([size[1],size[2],size[3]])
    extract.SetIndex([beginning[1],beginning[2],beginning[3]])
    extracted_image = extract.Execute(sitk_image)
    return extracted_image
end#sitk_crop


function test_single_crop(medIm::MedImage,sitk_image, begining, size)
    #sitk implementation
    cropped=sitk_crop(sitk_image,begining, size)
    
    #our Julia implementation
    medIm=crop_mi([medIm],begining,size,linear)[0]

    test_object_equality(medIm,cropped)

end #test_single_rotation   


function test_crops(path_nifti)    
    """
    test if the cropping of the image lead to correct change in the pixel array
    and the metadata the operation will be tasted against Python simple itk function

    """   
    for begining in [(10,11,13),(15,17,19)]
        for size in [(10,11,13),(15,17,19),(30,31,32)]
            medIm=load_image(path_nifti)
            sitk_image=sitk.ReadImage(path_nifti)
            test_single_crop(medIm,sitk_image, begining, size)
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

function test_pads(path_nifti)    
    """
    test if the padding of the image lead to correct change in the pixel array
    and the metadata the operation will be tasted against Python simple itk function

    """   
    for pad_beg in [(10,11,13),(15,17,19)]
        for pad_end in [(10,11,13),(15,17,19),(30,31,32)]
            for pad_val in [0.0,111.5]
                medIm=load_image(path_nifti)
                sitk_image=sitk.ReadImage(path_nifti)
                sitk_padded=sitk_pad(sitk_image, pad_beg, pad_end,pad_val)
                mi_padded=pad_mi(medIm,pad_beg,pad_end,pad_val,linear)
                test_object_equality(mi_padded,sitk_padded)
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
    reference_image = image 
    extracted_image=sitk.Resample(image, reference_image, transform,
    sitk.sitkLinear, 0.0)
    
    return extracted_image
end#sitk_translate




"""
test if the translation of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function

"""
function test_translate(path_nifti)
    
    # Load the image from path

    for t_val in [1,10,16]
        for axis in [1,2,3]
            medIm=load_image(path_nifti)
            sitk_image=sitk.ReadImage(path_nifti)
            sitk_trnanslated=sitk_translate(sitk_image,t_val, axis-1)
            medIm=translate_mi(medIm,t_val, axis,linear)
            test_object_equality(medIm,sitk_trnanslated)
        end#for    
    end#for

    

end
################################################# scaling tests




"""
test if the scaling of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function

"""
function test_scale(path_nifti)
    
    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)


    
    TODO()


    #save both images into nifti files to temporary folder
    

    test_image_equality(path_a,path_b)


end





# imagePath="/workspaces/MedImage.jl/test_data/volume-0.nii.gz"
# image = sitk.ReadImage(imagePath)
# cropped=sitk_translate(image,5,1)

# cropped.GetOrigin()
# image.GetOrigin()

# orig_arr,orig_spacing=getPixelsAndSpacing(image)
# cropped_arr,cropped_spacing=getPixelsAndSpacing(cropped)
# size(orig_arr)
# size(cropped_arr)

# disp_images(orig_arr,cropped_arr,(1.0,1.0,1.0))










# imagePath="/workspaces/MedImage.jl/test_data/volume-0.nii.gz"
# image = sitk.ReadImage(imagePath)


# cropped=sitk_pad(image,(10,15,17),(20,22,23),1.0)

# cropped.GetOrigin()
# image.GetOrigin()

# orig_arr,orig_spacing=getPixelsAndSpacing(image)
# cropped_arr,cropped_spacing=getPixelsAndSpacing(cropped)
# size(orig_arr)
# size(cropped_arr)
# rotated=rotation3d(image,2, 30)
# unrotated=rotation3d(rotated, 2,-30)

# rotated.GetOrigin()
# image.GetOrigin()

# rotated.GetDirection()
# image.GetDirection()
# rotated_arr,rotated_spacing=getPixelsAndSpacing(rotated)
# orig_arr,orig_spacing=getPixelsAndSpacing(image)
# unrotated_arr,unrotated_spacing=getPixelsAndSpacing(unrotated)

# rotated_arr=Float32.(rotated_arr)
# orig_arr=Float32.(orig_arr)
# unrotated_arr=Float32.(unrotated_arr)

