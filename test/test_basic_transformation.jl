"""
some issues that may occur and also why python rotation implementation is so clumsy here 
is described in https://stackoverflow.com/questions/56171643/simpleitk-rotation-of-volumetric-data-e-g-mri
we have separate python function fro z rotation and rotations in other planes
"""

include("../src/MedImage_data_struct.jl")
include("./test_visualize.jl")

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
    print(euler_transform.GetMatrix())
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

imagePath="/workspaces/MedImage.jl/test_data/volume-0.nii.gz"
image = sitk.ReadImage(imagePath)

rotated=rotation3d(image,2, 30)
unrotated=rotation3d(rotated, 1,270)




rotated_arr,rotated_spacing=getPixelsAndSpacing(rotated)
orig_arr,orig_spacing=getPixelsAndSpacing(image)
unrotated_arr,unrotated_spacing=getPixelsAndSpacing(unrotated)

rotated_arr=Float32.(rotated_arr)
orig_arr=Float32.(orig_arr)
unrotated_arr=Float32.(unrotated_arr)

disp_images(orig_arr,rotated_arr,(1.0,1.0,1.0))


"""
test if the rotation of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function

"""
function test_rotation(path_nifti)
    
    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)
 

    #series of values to rotate the image by
    
    TODO()
    rotation_vals=[[]]
    #rotate the image


    #save both images into nifti files to temporary folder
    

    test_image_equality(path_a,path_b)


end

"""
test if the cropping of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function

"""
function test_crop(path_nifti)
    
    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)


    
    TODO()


    #save both images into nifti files to temporary folder
    

    test_image_equality(path_a,path_b)


end


"""
test if the translation of the image lead to correct change in the pixel array
and the metadata the operation will be tasted against Python simple itk function

"""
function test_translate(path_nifti)
    
    # Load the image from path
    med_im=load_image(path_nifti)
    sitk.ReadImage(path_nifti)


    
    TODO()


    #save both images into nifti files to temporary folder
    

    test_image_equality(path_a,path_b)


end


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