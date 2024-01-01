using NIfTI,LinearAlgebra,DICOM
using .dicom_nifti
using PythonCall

# using CondaPkg
# CondaPkg.add("simpleitk")
# CondaPkg.add("numpy")
# CondaPkg.add_pip("simpleitk", version="")


sitk = pyimport("SimpleITK")
np = pyimport("numpy")


function get_center(img)
    """
    from python to test
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((Py(pyconvert(Int,np.ceil(width/2))), Py(pyconvert(Int,np.ceil(height/2)))
    , Py(pyconvert(Int,np.ceil(depth/2)))))
end #get_center


function rotation3d(image, theta_x, theta_y, theta_z)
    """
    from python to test
    """
    theta_x = np.deg2rad(theta_x)
    theta_y = np.deg2rad(theta_y)
    theta_z = np.deg2rad(theta_z)
    euler_transform = sitk.Euler3DTransform(get_center(image), theta_x, theta_y, theta_z, (0, 0, 0))
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)
    euler_transform.SetRotation(theta_x, theta_y, theta_z)
    resampled_image = sitk.Resample(image, euler_transform)
    return resampled_image
end #rotation3d


imagePath="/workspaces/MedImage.jl/test_data/volume-0.nii.gz"
image = sitk.ReadImage(imagePath)


rotated=rotation3d(image, 90, 0, 0)
unrotated=rotation3d(image, 270, 0, 0)

image_arr=pyconvert(Array, sitk.GetArrayFromImage(image))
unrotated_arr=pyconvert(Array,sitk.GetArrayFromImage(unrotated))

isapprox(image_arr, unrotated_arr, atol=10000.0)

image_arr[30,100,100]
unrotated_arr[30,100,100]


(Py(Int(pyconvert(np.ceil(5/2)))))
pyconvert(Int,np.ceil(5/2))

origSize = pyconvert(Array, image.GetSize())

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