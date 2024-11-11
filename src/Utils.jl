module Utils

using ..MedImage_data_struct
using KernelAbstractions, Interpolations

export interpolate_point
export get_base_indicies_arr
export cast_to_array_b_type
export interpolate_my
export TransformIndexToPhysicalPoint_julia
export ensure_tuple
export create_nii_from_medimage
import ..MedImage_data_struct: MedImage, Interpolator_enum, Mode_mi, Orientation_code, Nearest_neighbour_en, Linear_en, B_spline_en


"""
return array of cartesian indices for given dimensions in a form of array
"""
function get_base_indicies_arr(dims)
    indices = CartesianIndices(dims)
    # indices=collect.(Tuple.(collect(indices)))
    indices=Tuple.(collect(indices))
    indices=collect(Iterators.flatten(indices))
    indices=reshape(indices,(3,dims[1]*dims[2]*dims[3]))
    # indices=permutedims(indices,(1,2))
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
            a = round.(a)
        end

        a = convert(Array{eltype(b)}, a)
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


    if(keep_begining_same)
        if((i<1))
            i=1
        end
        if((j<1))
            j=1
        end
        if((k<1))
            k=1
        end
    end
    return itp(i, j,k)
    
end#interpolate_point    




@kernel function interpolate_kernel(out_res,@Const(source_arr_shape),@Const(source_arr),@Const(points_to_interpolate)
    ,@Const(spacing),keep_begining_same
    ,extrapolate_value,is_nearest_neighbour)
    shared_arr=@localmem(Float32, (512,3))


    index_local = @index(Local, Linear)
    I = @index(Global)

    shared_arr[index_local,1]=points_to_interpolate[1,I]
    shared_arr[index_local,2]=points_to_interpolate[2,I]
    shared_arr[index_local,3]=points_to_interpolate[3,I]

    #check for extrapolation - if point is outside of the image
    if(shared_arr[index_local,1]<0 || shared_arr[index_local,2]<0 || shared_arr[index_local,3]<0   || shared_arr[index_local,1]>=source_arr_shape[1] || shared_arr[index_local,2]>=source_arr_shape[2] || shared_arr[index_local,3]>=source_arr_shape[3])
        out_res[I]=extrapolate_value
    else
        if(shared_arr[index_local,1]<1  && keep_begining_same)
            shared_arr[index_local,1]=1
        end
        if(shared_arr[index_local,2]<1 && keep_begining_same)
            shared_arr[index_local,2]=1
        end    
        if(shared_arr[index_local,3]<1  && keep_begining_same)
            shared_arr[index_local,3]=1
        end    
        if(shared_arr[index_local,1]==1 || shared_arr[index_local,2]==2 || shared_arr[index_local,3]==3)
            out_res[I]=source_arr[Int(round(shared_arr[index_local,1])),Int(round(shared_arr[index_local,2])),Int(round(shared_arr[index_local,3]))]
        else
            #simple implementation of nearest neighbour
            if(is_nearest_neighbour)
                #check x axis - if we are closer to the lower or upper bound
                if( abs(((shared_arr[index_local,1]-floor( shared_arr[index_local,1]))*spacing[1])) < abs(((shared_arr[index_local,1]-ceil( shared_arr[index_local,1])))*spacing[1]))
                    #floor x smaller
                    shared_arr[index_local,1]=floor( shared_arr[index_local,1])
                else
                    #ceil x smaller
                    shared_arr[index_local,1]=ceil( shared_arr[index_local,1])
                end
                #check y axis - if we are closer to the lower or upper bound
                if( abs(((shared_arr[index_local,2]-floor( shared_arr[index_local,2]))*spacing[2])) < abs(((shared_arr[index_local,2]-ceil( shared_arr[index_local,2])))*spacing[2]))
                    #floor y smaller
                    shared_arr[index_local,2]=floor( shared_arr[index_local,2])
                else
                    #ceil y smaller
                    shared_arr[index_local,2]=ceil( shared_arr[index_local,2])
                end
                #check z axis - if we are closer to the lower or upper bound
                if( abs(((shared_arr[index_local,3]-floor( shared_arr[index_local,3]))*spacing[3])) < abs(((shared_arr[index_local,3]-ceil( shared_arr[index_local,3]))*spacing[3])))
                    #floor z smaller
                    shared_arr[index_local,3]=floor( shared_arr[index_local,3])
                else
                    #ceil z smaller
                    shared_arr[index_local,3]=ceil( shared_arr[index_local,3])
                end

                #we ha already selected x,y,z so we can return the value
                out_res[I]=source_arr[Int(shared_arr[index_local,1]),Int(shared_arr[index_local,2]),Int(shared_arr[index_local,3])]
            else
                # if we are not using nearest neighbour we will use linear interpolation 
                #interpolation based on distance of each point to the 8 points in the cube around it divided by recaluclated sum of those distances
                shared_arr[index_local,1]=(( (source_arr[Int(ceil( shared_arr[index_local,1])), Int(ceil( shared_arr[index_local,2])), Int(ceil( shared_arr[index_local,3]))]*( (1/sqrt( ((((shared_arr[index_local,1]-ceil( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-ceil( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-ceil( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ))   +  (source_arr[Int(ceil( shared_arr[index_local,1])), Int(ceil( shared_arr[index_local,2])), Int(floor( shared_arr[index_local,3]))]*( (1/sqrt( ((((shared_arr[index_local,1]-ceil( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-ceil( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-floor( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ))   +  (source_arr[Int(ceil( shared_arr[index_local,1])), Int(floor( shared_arr[index_local,2])), Int(ceil( shared_arr[index_local,3]))]*( (1/sqrt( ((((shared_arr[index_local,1]-ceil( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-floor( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-ceil( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ))   +  (source_arr[Int(ceil( shared_arr[index_local,1])), Int(floor( shared_arr[index_local,2])), Int(floor( shared_arr[index_local,3]))]*( (1/sqrt( ((((shared_arr[index_local,1]-ceil( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-floor( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-floor( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ))   +  (source_arr[Int(floor( shared_arr[index_local,1])), Int(ceil( shared_arr[index_local,2])), Int(ceil( shared_arr[index_local,3]))]*( (1/sqrt( ((((shared_arr[index_local,1]-floor( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-ceil( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-ceil( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ))   +  (source_arr[Int(floor( shared_arr[index_local,1])), Int(ceil( shared_arr[index_local,2])), Int(floor( shared_arr[index_local,3]))]*( (1/sqrt( ((((shared_arr[index_local,1]-floor( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-ceil( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-floor( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ))   +  (source_arr[Int(floor( shared_arr[index_local,1])), Int(floor( shared_arr[index_local,2])), Int(ceil( shared_arr[index_local,3]))]*( (1/sqrt( ((((shared_arr[index_local,1]-floor( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-floor( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-ceil( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ))   +  (source_arr[Int(floor( shared_arr[index_local,1])), Int(floor( shared_arr[index_local,2])), Int(floor( shared_arr[index_local,3]))]*( (1/sqrt( ((((shared_arr[index_local,1]-floor( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-floor( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-floor( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ))  )/(( (1/sqrt( ((((shared_arr[index_local,1]-ceil( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-ceil( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-ceil( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ) + ( (1/sqrt( ((((shared_arr[index_local,1]-ceil( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-ceil( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-floor( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ) + ( (1/sqrt( ((((shared_arr[index_local,1]-ceil( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-floor( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-ceil( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ) + ( (1/sqrt( ((((shared_arr[index_local,1]-ceil( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-floor( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-floor( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ) + ( (1/sqrt( ((((shared_arr[index_local,1]-floor( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-ceil( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-ceil( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ) + ( (1/sqrt( ((((shared_arr[index_local,1]-floor( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-ceil( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-floor( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ) + ( (1/sqrt( ((((shared_arr[index_local,1]-floor( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-floor( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-ceil( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) ) + ( (1/sqrt( ((((shared_arr[index_local,1]-floor( shared_arr[index_local,1]))*spacing[1])^2)
                    +(((shared_arr[index_local,2]-floor( shared_arr[index_local,2]))*spacing[2])^2)
                    + (((shared_arr[index_local,3]-floor( shared_arr[index_local,3]))*spacing[3])^2) )+0.000001
                    )) )))


                out_res[I]=shared_arr[index_local,1]
            end#is one 
        end#is_nearest_neighbour
    end#check is in range    
end




"""
perform the interpolation of the set of points in a given space
input_array - array we will use to find interpolated val
input_array_spacing - spacing associated with array from which we will perform interpolation
Interpolator_enum - enum value defining the type of interpolation
keep_begining_same - will keep unmodified first layer of each axis - usefull when changing spacing
extrapolate_value - value to use for extrapolation

IMPORTANT!!! - by convention if index to interpolate is less than 0 we will use extrapolate_value (we work only on positive indicies here)
"""
function interpolate_my(points_to_interpolate,input_array,input_array_spacing,interpolator_enum,keep_begining_same, extrapolate_value=0,use_fast=true)

    old_size=size(input_array)
    interpolator_enum=Int(interpolator_enum)
    interpolator_enum=instances(Interpolator_enum)[interpolator_enum+1]

    if(use_fast)
        is_nearest_neighbour=(interpolator_enum == Nearest_neighbour_en)
        out_res = similar(points_to_interpolate, eltype(points_to_interpolate), size(points_to_interpolate, 2))
        backend = get_backend(points_to_interpolate)
        source_arr_shape=size(input_array)
        interpolate_kernel(backend, 512)(out_res,source_arr_shape,input_array,points_to_interpolate
        ,input_array_spacing,keep_begining_same
        ,extrapolate_value,is_nearest_neighbour, ndrange=size(out_res))
        synchronize(backend)
        return out_res

    end
    #if we do not want to use fast version we will use the slower one but more flexible
    if interpolator_enum == Nearest_neighbour_en
        itp = interpolate(input_array, BSpline(Constant()))
    elseif interpolator_enum == MedImage_data_struct.Linear_en
        itp = interpolate(input_array, BSpline(Linear()))
    elseif interpolator_enum == MedImage_data_struct.B_spline_en
        itp = interpolate(input_array, BSpline(Cubic(Line(OnGrid()))))
    end

    #we indicate on each axis the spacing from area we are samplingA
    # A_x1 = 1:input_array_spacing[1]:(old_size[1]+input_array_spacing[1]*old_size[1])
    # A_x2 = 1:input_array_spacing[2]:(old_size[2]+input_array_spacing[2]*old_size[2])
    # A_x3 = 1:input_array_spacing[3]:(old_size[3]+input_array_spacing[3]*old_size[3])

    A_x1 = 1:input_array_spacing[1]:(1 + input_array_spacing[1] * (old_size[1] - 1))
    A_x2 = 1:input_array_spacing[2]:(1 + input_array_spacing[2] * (old_size[2] - 1))
    A_x3 = 1:input_array_spacing[3]:(1 + input_array_spacing[3] * (old_size[3] - 1))


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

# ############ fast interpolation

# """
# calculate distance between set location and neighbouring voxel coordinates
# """
# macro get_dist(a,b,c)
#   return  esc(:(
#   sqrt(((((shared_arr[index_loc,1]-round(shared_arr[index_loc,1]+($a) ) *input_array_spacing[1])  )^2
#   +((shared_arr[index_loc,2]-round(shared_arr[index_loc,2] +($b)))*input_array_spacing[2])^2
#   +((shared_arr[index_loc,3]-round(shared_arr[index_loc,3]+($c)))*input_array_spacing[3])^2)^2  )+0.00000001)#we add a small number to avoid division by 0
# ))
# end

# """
# get interpolated value at point
# """
# macro get_interpolated_val(input_array,a,b,c)

#   return  esc(quote
#       var3=(@get_dist($a,$b,$c))
#       var1+=var3
#       ($input_array[Int(round(shared_arr[index_loc,1]+($a)))
#       ,Int(round(shared_arr[index_loc,2]+($b)))
#       ,Int(round(shared_arr[index_loc,3]+($c)))]/(var3 ))

#     end)
# end

# """
# used to get approximation of local variance
# """
# macro get_interpolated_diff(input_array,a,b,c)

#   return  esc(quote
#       ((($input_array[Int(round(shared_arr[index_loc,1]+($a)))
#       ,Int(round(shared_arr[index_loc,2]+($b)))
#       ,Int(round(shared_arr[index_loc,3]+($c)))])-var2)^2)*((@get_dist($a,$b,$c)) )
# end)
# end

# """
# simple kernel friendly interpolator - given float coordinates and source array will
# 1) look for closest integers in all directions and calculate the euclidean distance to it
# 2) calculate the weights for each of the 27 points in the cube around the pointadding more weight the closer the point is to integer coordinate
# we take into account spacing
# """
# macro threeDLinInterpol(input_array)
#   ## first we get the total distances of all points to be able to normalize it later
#   return  esc(quote
#   var1=0.0#
#   var2=0.0
#   var3=0.0
#   for a1 in -0.5:0.5:0.5
#      for b1 in -0.5:0.5:0.5
#        for c1 in -0.5:0.5:0.5
#         if((shared_arr[index_loc,1]+a1)>0 || (shared_arr[index_loc,2]+b1)>0 || (shared_arr[index_loc,3]+c1)>0
#             || (shared_arr[index_loc,1]+a1)<input_array_size[1] || (shared_arr[index_loc,2]+b1)<input_array_size[2] || (shared_arr[index_loc,3]+c1)<input_array_size[3])
#             var2+= @get_interpolated_val(input_array,a1,b1,c1)
#         end
#         end
#       end
#   end
#   var2=var2/var1
#   end)
# end

# """
# version that takes into account wider context of the point
# """
# macro threeDLinInterpol_wider(input_array)
#     ## first we get the total distances of all points to be able to normalize it later
#     return  esc(quote
#     var1=0.0#
#     var2=0.0
#     var3=0.0
#     for a1 in -1.5:1.5:0.5
#        for b1 in -1.5:1.5:0.5
#          for c1 in -1.5:1.5:0.5
#                 if((shared_arr[index_loc,1]+a1)>0 || (shared_arr[index_loc,2]+b1)>0 || (shared_arr[index_loc,3]+c1)>0
#                     || (shared_arr[index_loc,1]+a1)<input_array_size[1] || (shared_arr[index_loc,2]+b1)<input_array_size[2] || (shared_arr[index_loc,3]+c1)<input_array_size[3])


#                     var2+= @get_interpolated_val(input_array,a1,b1,c1)
#                 end
#             end
#         end
#     end
#     var2=var2/var1
#     end)
#   end



# @kernel function interpolate_point_fast_linear(points_arr,input_array, input_array_spacing,points_out,input_array_size,keep_begining_same,extrapolate_value)
#     index_glob= @index(Global, Linear)
#     index_loc=@index(Local, Linear)
#     # +1 to avoid bank conflicts on shared memory
#     shared_arr = @localmem eltype(points_arr) ((@uniform @groupsize()[1])+1, 3)
#     #copy from global to local memory
#     shared_arr[index_loc,1]=points_arr[1,index_glob]
#     shared_arr[index_loc,2]=points_arr[2,index_glob]
#     shared_arr[index_loc,3]=points_arr[3,index_glob]
#     #check for extrapolation
#     if(shared_arr[index_loc,1]<0 || shared_arr[index_loc,2]<0 || shared_arr[index_loc,3]<0 || shared_arr[index_loc,1]>input_array_size[1] || shared_arr[index_loc,2]>input_array_size[2] || shared_arr[index_loc,3]>input_array_size[3])
#         points_out[index_glob]=extrapolate_value
#     else
#         if(keep_begining_same && ((shared_arr[index_loc,1]<1)))
#             shared_arr[index_loc,1]=1
#         end
#         if(keep_begining_same && ((shared_arr[index_loc,2]<1)))
#             shared_arr[index_loc,2]=1
#         end
#         if(keep_begining_same && ((shared_arr[index_loc,3]<1)))
#             shared_arr[index_loc,3]=1
#         end

#         #initialize variables in registry
#         var1=0.0
#         var2=0.0
#         var3=0.0
#         @threeDLinInterpol(input_array)
#         #save output
#         points_out[index_glob]=var2
#     end

# end

# function call_interpolate_point_fast_linear(points_arr,input_array, input_array_spacing,keep_begining_same=false,extrapolate_value=0)
#     points_out=similar(points_arr)

#     dev = get_backend(points_arr)
#     interpolate_point_fast_linear(dev, 256)(points_arr,input_array, input_array_spacing,points_out,input_array_size,keep_begining_same, ndrange=(size(points_arr)[2]))
#     KernelAbstractions.synchronize(dev)
#     return points_out

# end


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
#         MRI,  # Założenie, że Image_type to Enum z wartości�� MRI jako domyślną
#         subtypes,  # Założenie, że Image_subtype to Enum z wartością subtypes jako domyślną; upewnij się, że to ma sens w kontekście twojego kodu
#         typeof(1.0),  # Przykład typu danych; dostosuj do swoich potrzeb
#         "", "", "", "CPU", "", "", "", "", "", "", Dictionary(), false, Dictionary()
#       )
#     return med_image
# end


"""
create_nii_from_medimage(med_image::MedImage, file_path::String)

Create a .nii.gz file from a MedImage object and save it to the given file path.
"""
function create_nii_from_medimage(med_image::MedImage, file_path::String)
    sitk = pyimport("SimpleITK")
    np = pyimport("numpy")
    voxel_data_np = np.array(med_image.voxel_data)

    image_sitk = sitk.GetImageFromArray(voxel_data_np)
    image_sitk.SetOrigin(med_image.origin)
    image_sitk.SetSpacing(med_image.spacing)
    image_sitk.SetDirection(med_image.direction)

    sitk.WriteImage(image_sitk, file_path * ".nii.gz")
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


end#Utils
