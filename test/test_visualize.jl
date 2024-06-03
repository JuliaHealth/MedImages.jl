# # using PythonCall
# import MedEye3d
# import MedEye3d.ForDisplayStructs
# import MedEye3d.ForDisplayStructs.TextureSpec
# using ColorTypes
# import MedEye3d.SegmentationDisplay

# import MedEye3d.DataStructs.ThreeDimRawDat
# import MedEye3d.DataStructs.DataToScrollDims
# import MedEye3d.DataStructs.FullScrollableDat
# import MedEye3d.ForDisplayStructs.KeyboardStruct
# import MedEye3d.ForDisplayStructs.MouseStruct
# import MedEye3d.ForDisplayStructs.ActorWithOpenGlObjects
# import MedEye3d.OpenGLDisplayUtils
# import MedEye3d.DisplayWords.textLinesFromStrings
# import MedEye3d.StructsManag.getThreeDims
# import MedEye3d.DisplayWords.textLinesFromStrings
# import MedEye3d.StructsManag.getThreeDims


# """
# designed to visualize two arrays in the same window
# """


# """
# becouse Julia arrays is column wise contiguus in memory and open GL expects row wise we need to rotate and flip images 
# pixels - 3 dimensional array of pixel data 
# """
# function permuteAndReverse(pixels)
#     pixels=  permutedims(pixels, (3,2,1))
#     sizz=size(pixels)
#     for i in 1:sizz[1]
#         for j in 1:sizz[3]
#             pixels[i,:,j] =  reverse(pixels[i,:,j])
#         end# 
#     end# 
#     return pixels
#   end#permuteAndReverse

# """
# given simple ITK image it reads associated pixel data - and transforms it by permuteAndReverse functions
# it will also return voxel spacing associated with the image
# """
# function getPixelsAndSpacing(image)
#     pixelsArr = pyconvert(Array,sitk.GetArrayFromImage(image))# we need numpy in order for pycall to automatically change it into julia array
#     spacings = image.GetSpacing()
#     return ( permuteAndReverse(pixelsArr), pyconvert(Tuple{Float64, Float64, Float64},spacings)  )
# end#getPixelsAndSpacing



# # image_arr=pyconvert(Array, sitk.GetArrayFromImage(image))
# # unrotated_arr=pyconvert(Array,sitk.GetArrayFromImage(unrotated))
# listOfTexturesSpec = [
#   TextureSpec{Float32}(
#       name = "rot",
#       isMainImage = true,
#       minAndMaxValue= Int16.([0,100])
#      ),
#   TextureSpec{Float32}(
#       name = "inrot",
#       color = RGB(0.0,1.0,0.0)
#       ,minAndMaxValue= Float32.([-10,100])
#       ,isEditable = true
#      ),
#      TextureSpec{UInt8}(
#         name = "manualModif",
#         color = RGB(0.0,1.0,0.0)
#         ,minAndMaxValue= UInt8.([0,1])
#         ,isEditable = true
#        ),
# ];

# """
# function that will display arra and shadow of arr2 on top of it to compare ThreeDimRawDat
#     both arrays should have the same dimensions and spacing
# """
# function disp_images(arr_a,arr_b,spacing)
#     mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
#     supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(arr_a)[3] );


#     tupleVect = [("rot",arr_a) ,("inrot",arr_b),("manualModif",zeros(UInt8,size(arr_b))) ]
#     # tupleVect = [("rot",unrotated_arr) ,("inrot",rotated_arr),("manualModif",zeros(UInt8,size(unrotated_arr))) ]
#     slicesDat= getThreeDims(tupleVect )
    
#     # datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(rotated_arr) ,voxelSize=(rotated_spacing[3],rotated_spacing[2],rotated_spacing[1]), dimensionToScroll = 3 );
#     datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(arr_a) ,voxelSize=spacing, dimensionToScroll = 3 );
    
#     mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
#                                      ,dimensionToScroll=3 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
#                                      ,dataToScroll= slicesDat
#                                      ,mainTextToDisp= mainLines
#                                      ,sliceTextToDisp=supplLines );
    
#     fractionOfMainIm= Float32(0.8);
#     SegmentationDisplay.coordinateDisplay(listOfTexturesSpec ,fractionOfMainIm ,datToScrollDimsB ,1000);
#     Main.SegmentationDisplay.passDataForScrolling(mainScrollDat);
#     return mainScrollDat



# end