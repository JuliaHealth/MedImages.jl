
"""
return array of cartesian indices for given dimensions in a form of array
"""
function get_base_indicies_arr(dims)    
    indices = CartesianIndices(dims)
    # indices=collect.(Tuple.(collect(indices)))
    indices=Tuple.(collect(indices))
    indices=collect(Iterators.flatten(indices))
    indices=reshape(indices,(3,dims[1]*dims[2]*dims[3]))
    indices=permutedims(indices,(1,2))
    return indices
  end#get_base_indicies_arr
  
"""
cast array a to the value type of array b
"""
function cast_to_array_b_type(a,b)
    # Check if array a and b have the same type
    if eltype(a) != eltype(b)
        # Cast array a to the value type of array b
        if eltype(b) in [Int8,Int16,Int32, Int64, UInt8,UInt16,UInt32, UInt64]
            # Apply rounding to the array
            a = round.(a)  # Array{Int64,1}
        end

        a = convert(Array{eltype(b)}, a)  # Array{Float64,1}
        return a
    else
        return a
    end
end

"""
interpolate the point in the given space
keep_begining_same - will keep unmodified first layer of each axis - usefull when changing spacing
"""
function interpolate_point(point,itp, keep_begining_same=false,extrapolate_value=0)

    i=point[1]
    j=point[2]
    k=point[3]
    if(i<0 || j<0 || k<0)
        return extrapolate_value
    end

    i_1= max(i,1)
    j_1= max(j,1)
    k_1= max(k,1)
    if(keep_begining_same)
        if((i==1))
            i_1=1
        end        
        if((j==1))
            j_1=1
        end        
        if((k==1))
            k_1=1
        end                
    end
    return itp(i_1, j_1,k_1)
    
end#interpolate_point    

"""
perform the interpolation of the set of points in a given space
input_array - array we will use to find interpolated val
input_array_spacing - spacing associated with array from which we will perform interpolation
Interpolator_enum - enum value defining the type of interpolation
keep_begining_same - will keep unmodified first layer of each axis - usefull when changing spacing
extrapolate_value - value to use for extrapolation

IMPORTANT!!! - by convention if index to interpolate is less than 0 we will use extrapolate_value (we work only on positive indicies here)
"""
function interpolate_my(points_to_interpolate,input_array,input_array_spacing,interpolator_enum,keep_begining_same, extrapolate_value=0)

    old_size=size(input_array)
    if interpolator_enum == Nearest_neighbour_en
        itp = interpolate(input_array, BSpline(Constant()))
    elseif interpolator_enum == Linear_en
        itp = interpolate(input_array, BSpline(Linear()))
    elseif interpolator_enum == B_spline_en
        itp = interpolate(input_array, BSpline(Cubic(Line(OnGrid()))))
    end
    #we indicate on each axis the spacing from area we are samplingA
    A_x1 = 1:input_array_spacing[1]:(old_size[1]+input_array_spacing[1]*old_size[1])
    A_x2 = 1:input_array_spacing[2]:(old_size[2]+input_array_spacing[2]*old_size[2])
    A_x3 = 1:input_array_spacing[3]:(old_size[3]+input_array_spacing[3]*old_size[3])
    
    itp=extrapolate(itp, extrapolate_value)   
    itp = scale(itp, A_x1, A_x2,A_x3)
    # Create the new voxel data
    # print("eeeeeeeeeeeeee $(itp(-1222.0,-1222.0,-1222.0))")


    res=collect(range(1,size(points_to_interpolate)[2]))
    res=map(el->interpolate_point( points_to_interpolate[:,el],itp,keep_begining_same,extrapolate_value),res)

    return res
end#interpolate_my

function TransformIndexToPhysicalPoint_julia(index::Tuple{Int,Int,Int}
    ,origin::Tuple{Float64,Float64,Float64}
    ,spacing::Tuple{Float64,Float64,Float64})
    
    # return origin .+ ((collect(index) .- 1) .* collect(spacing))
    return collect(collect(origin) .+ ((collect(index) ) .* collect(spacing)))
end



function ensure_tuple(arr)
    if arr isa Tuple
        return arr
    elseif arr isa AbstractArray
        return tuple(arr...)
    else
        error("Input must be a Tuple or an AbstractArray")
    end
  end

# path_nifti="/home/jakubmitura/projects/MedImage.jl/test_data/volume-0.nii.gz"
# im=sitk.ReadImage(path_nifti)
# indexx=(2,2,2)
# function TransformIndexToPhysicalPoint_julia(index::Tuple{Int,Int,Int}
#     ,origin::Tuple{Float64,Float64,Float64}
#     ,spacing::Tuple{Float64,Float64,Float64})
#     # return origin .+ ((collect(index) .- 1) .* collect(spacing))
#     return collect(collect(origin) .+ ((collect(index) ) .* collect(spacing)))
# end

# path_nifti="/home/jakubmitura/projects/MedImage.jl/test_data/volume-0.nii.gz"
# im=sitk.ReadImage(path_nifti)
# indexx=(1,1,1)
# im.TransformIndexToPhysicalPoint(indexx)
# med_im = load_image(path_nifti)
# TransformIndexToPhysicalPoint_julia(indexx,med_im.origin,med_im.spacing)


# im.GetOrigin()
# med_im.origin

# (-172.19686889648438, 178.59375, -363.0)


# im.TransformIndexToPhysicalPoint(indexx)
# orient_filter=sitk.DICOMOrientImageFilter()
# orient_filter.SetDesiredCoordinateOrientation("LAS")
# oriented_image = orient_filter.Execute(im)
# oriented_image.TransformIndexToPhysicalPoint(indexx)

# a



# sitk = pyimport_conda("SimpleITK", "simpleitk")
# np = pyimport("numpy")

# """
# get_spatial_metadata(image_path::String)::MedImage

# Funkcja wczytuje obraz z podanej ścieżki i ekstrahuje jego podstawowe metadane przestrzenne.
# Zwraca obiekt MedImage zawierający te metadane oraz dane obrazu.
# """
# function get_spatial_metadata(image_path::String)::MedImage
#     image = sitk.ReadImage(image_path)
#     origin = image.GetOrigin()
#     spacing = image.GetSpacing()
#     direction = image.GetDirection()
#     voxel_data = sitk.GetArrayFromImage(image)
   
#     med_image = MedImage(
#         voxel_data,  # Przykład pustej wielowymiarowej tablicy
#         origin,  # Pusty Tuple dla origin
#         spacing,  # Pusty Tuple dla spacing
#         direction,  # Pusty Tuple dla direction
#         Dictionary(),  # Pusty słownik
#         MRI,  # Założenie, że Image_type to Enum z wartością MRI jako domyślną
#         subtypes,  # Założenie, że Image_subtype to Enum z wartością subtypes jako domyślną; upewnij się, że to ma sens w kontekście twojego kodu
#         typeof(1.0),  # Przykład typu danych; dostosuj do swoich potrzeb
#         "", "", "", "CPU", "", "", "", "", "", "", Dictionary(), false, Dictionary()
#       )
#     return med_image
# end


# """
# create_nii_from_medimage(med_image::MedImage, file_path::String)

# Create a .nii.gz file from a MedImage object and save it to the given file path.
# """
function create_nii_from_medimage(med_image::MedImage, file_path::String)
    # Convert voxel_data to a numpy array (Assuming voxel_data is stored in Julia array format)
    voxel_data_np = np.array(med_image.voxel_data)
    
    # Create a SimpleITK image from numpy array
    image_sitk = sitk.GetImageFromArray(voxel_data_np)
    
    # Set spatial metadata
    image_sitk.SetOrigin(med_image.origin)
    image_sitk.SetSpacing(med_image.spacing)
    image_sitk.SetDirection(med_image.direction)
    
    # Save the image as .nii.gz
    sitk.WriteImage(image_sitk, file_path* ".nii.gz")
end


# function update_voxel_data(old_image::MedImage, new_voxel_data::AbstractArray)
  
#     return MedImage(
#         new_voxel_data, 
#         old_image.origin, 
#         old_image.spacing, 
#         old_image.direction, 
#         old_image.spatial_metadata, 
#         old_image.image_type, 
#         old_image.image_subtype, 
#         old_image.voxel_datatype, 
#         old_image.date_of_saving, 
#         old_image.acquistion_time, 
#         old_image.patient_id, 
#         old_image.current_device, 
#         old_image.study_uid, 
#         old_image.patient_uid, 
#         old_image.series_uid, 
#         old_image.study_description, 
#         old_image.legacy_file_name, 
#         old_image.display_data, 
#         old_image.clinical_data, 
#         old_image.is_contrast_administered, 
#         old_image.metadata)

# image_3D=get_spatial_metadata("C:\\MedImage\\MedImage.jl\\test_data\\volume-0.nii.gz")
# image_path_3D = "C:\\MedImage\\MedImage.jl\\test_data\\volume-0.nii.gz"
# image_test_3D = sitk.ReadImage(image_path_3D)
# image_4D=get_spatial_metadata("C:\\MedImage\\MedImage.jl\\test_data\\filtered_func_data.nii.gz")
# image_path_4D = "C:\\MedImage\\MedImage.jl\\test_data\\filtered_func_data.nii.gz"
# image_test_4D = sitk.ReadImage(image_path_4D)
# print(image.GetOrigin())
# print(image.GetSpacing())
# print(image.GetDirection())


