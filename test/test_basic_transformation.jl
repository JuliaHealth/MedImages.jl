using .dicom_nifti

using NIfTI,LinearAlgebra,DICOM
using PythonCall
import MedEye3d
import MedEye3d.ForDisplayStructs
import MedEye3d.ForDisplayStructs.TextureSpec
using ColorTypes
import MedEye3d.SegmentationDisplay

import MedEye3d.DataStructs.ThreeDimRawDat
import MedEye3d.DataStructs.DataToScrollDims
import MedEye3d.DataStructs.FullScrollableDat
import MedEye3d.ForDisplayStructs.KeyboardStruct
import MedEye3d.ForDisplayStructs.MouseStruct
import MedEye3d.ForDisplayStructs.ActorWithOpenGlObjects
import MedEye3d.OpenGLDisplayUtils
import MedEye3d.DisplayWords.textLinesFromStrings
import MedEye3d.StructsManag.getThreeDims

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
unrotated=rotation3d(rotated, 270, 0, 0)



"""
becouse Julia arrays is column wise contiguus in memory and open GL expects row wise we need to rotate and flip images 
pixels - 3 dimensional array of pixel data 
"""
function permuteAndReverse(pixels)
    pixels=  permutedims(pixels, (3,2,1))
    sizz=size(pixels)
    for i in 1:sizz[1]
        for j in 1:sizz[3]
            pixels[i,:,j] =  reverse(pixels[i,:,j])
        end# 
    end# 
    return pixels
  end#permuteAndReverse

"""
given simple ITK image it reads associated pixel data - and transforms it by permuteAndReverse functions
it will also return voxel spacing associated with the image
"""
function getPixelsAndSpacing(image)
    pixelsArr = pyconvert(Array,sitk.GetArrayFromImage(image))# we need numpy in order for pycall to automatically change it into julia array
    spacings = image.GetSpacing()
    return ( permuteAndReverse(pixelsArr), pyconvert(Tuple{Float64, Float64, Float64},spacings)  )
end#getPixelsAndSpacing


rotated_arr,rotated_spacing=getPixelsAndSpacing(rotated)
orig_arr,orig_spacing=getPixelsAndSpacing(rotated)
unrotated_arr,unrotated_spacing=getPixelsAndSpacing(unrotated)

rotated_arr=Float32.(rotated_arr)
orig_arr=Float32.(orig_arr)
unrotated_arr=Float32.(unrotated_arr)

# image_arr=pyconvert(Array, sitk.GetArrayFromImage(image))
# unrotated_arr=pyconvert(Array,sitk.GetArrayFromImage(unrotated))
listOfTexturesSpec = [
  TextureSpec{Float32}(
      name = "rot",
      isMainImage = true,
      minAndMaxValue= Int16.([0,100])
     ),
  TextureSpec{Float32}(
      name = "inrot",
      color = RGB(0.0,1.0,0.0)
      ,minAndMaxValue= Float32.([-10,100])
      ,isEditable = true
     ),
     TextureSpec{UInt8}(
        name = "manualModif",
        color = RGB(0.0,1.0,0.0)
        ,minAndMaxValue= UInt8.([0,1])
        ,isEditable = true
       ),
];
import MedEye3d.DisplayWords.textLinesFromStrings

mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(rotated_arr)[3] );


import MedEye3d.StructsManag.getThreeDims

tupleVect = [("rot",orig_arr) ,("inrot",orig_arr),("manualModif",zeros(UInt8,size(unrotated_arr))) ]
# tupleVect = [("rot",unrotated_arr) ,("inrot",rotated_arr),("manualModif",zeros(UInt8,size(unrotated_arr))) ]
slicesDat= getThreeDims(tupleVect )

# datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(rotated_arr) ,voxelSize=(rotated_spacing[3],rotated_spacing[2],rotated_spacing[1]), dimensionToScroll = 3 );
datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(rotated_arr) ,voxelSize=orig_spacing, dimensionToScroll = 3 );

mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
                                 ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
                                 ,dataToScroll= slicesDat
                                 ,mainTextToDisp= mainLines
                                 ,sliceTextToDisp=supplLines );

fractionOfMainIm= Float32(0.8);
SegmentationDisplay.coordinateDisplay(listOfTexturesSpec ,fractionOfMainIm ,datToScrollDimsB ,1000);
Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);


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