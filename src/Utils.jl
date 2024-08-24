
"""
`get_base_indicies_arr(dims::Tuple{Int, Int, Int})::Array{Int, 2}`

Extract and format Cartesian indices for a given dimensional space into a convenient array format.
This function is used to handle 3D array manipulations by converting multidimensional indices into a flat 2D array.

# Arguments
- `dims`: Dimensions of the 3D space as a tuple indicating the size in each dimension. 
 You can get them by using the size(Your_MedImage.voxel_data).

# Returns
- `Array{Int, 2}`: A 2D array where each column represents the flat index for each point in the 3D space.
"""
function get_base_indicies_arr(dims::Tuple{Int, Int, Int})::Array{Int, 2}    
    indices = CartesianIndices(dims) # Generate Cartesian indices for given dimensions.
    indices = Tuple.(collect(indices))
    indices = collect(Iterators.flatten(indices))  #Flatten the tuples into a single array.
    indices = reshape(indices,(3,dims[1]*dims[2]*dims[3])) # Reshape into a 2D array.
    return indices
end



"""
`cast_to_array_b_type(a, b)::Array`

Converts array `a` to the type of array `b`, applying rounding if necessary to prevent inexact error messages during type conversion, particularly useful when converting floating-point numbers to integers.
If `b` is of an integer type, `a` is rounded first to avoid casting errors.

# Arguments
- `a`: The array to be converted.
- `b`: The array whose type `a` is to be converted to.

# Returns
- `Array`: Array `a` converted to the type of array `b`.
"""
function cast_to_array_b_type(a::Array,b::Array)::Array
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
`interpolate_point(point, itp, keep_begining_same::Bool=false, extrapolate_value=0)::Float`

Interpolates a value at a specified point within a grid using a provided interpolation function. Negative indices are handled by returning an `extrapolate_value`, allowing for extrapolation outside the grid boundaries.
If any index in `point` is negative, the function returns `extrapolate_value`. If `keep_begining_same` is `true`, the function ensures that the point does not fall below the grid start.

# Arguments
- `point`: A tuple of indices (i, j, k) for which the value needs to be interpolated.
- `itp`: The interpolation function. To be correct, the function must be created via the prep_itp(), due to the lack of native support for spacing.
- `keep_begining_same`: If `true`, adjust the point to ensure it does not fall below the grid start (defaults to `false`).
- `extrapolate_value`: Value to return for out-of-bound indices (defaults to 0). Here you can use the average of the corners of the image - extrapolate_corner_median(), which should better simulate the real empty space

# Returns
- `Float`: The interpolated value at the given point.
"""
function interpolate_point(point, itp, keep_begining_same::Bool=false, extrapolate_value=0)

    i=point[1]
    j=point[2]
    k=point[3]

    # Return extrapolate value for negative indices.
    if(i<0 || j<0 || k<0)
        return extrapolate_value
    end

    # Optionally adjust indices to ensure they start from at least 1.
    if(keep_begining_same)
        i = max(i, 1)
        j = max(j, 1)
        k = max(k, 1)          
    end
    return itp(i, j,k) # Perform interpolation at the adjusted indices.
    
end  


"""
`interpolate_my(points_to_interpolate, input_array, input_array_spacing, interpolator_enum, keep_begining_same, extrapolate_value=0)::Array`

Interpolates a set of points within an input array using a predefined interpolation setup prepared by `prep_itp`. This function is designed to handle both interpolation with non-isovolumetric spacing adjustments, handling negative indices and extrapolation based on the grid defined by the `input_array_spacing`.
IMPORTANT!!! - by convention if index to interpolate is less than 0 we will use extrapolate_value (we work only on positive indicies here)

# Arguments
- `points_to_interpolate`: Array of points where interpolation is to be performed.
- `input_array`: The data array from which values are interpolated.
- `input_array_spacing`: Spacing adjustments for each dimension of `input_array`. You can get them by using the Your_MedImage.spacing method.
- `interpolator_enum`: The type of interpolation method to use (e.g., Nearest neighbor, Linear, B-spline).
- `keep_begining_same`: If `true`, ensure that interpolation does not go below the start of the grid.
- `extrapolate_value`: The value used for extrapolation, returned where interpolation is outside the bounds of `input_array`.

# Returns
- `Array`: The interpolated values at each point specified in `points_to_interpolate`.
"""
function interpolate_my(points_to_interpolate::Array, input_array::Array, input_array_spacing::Tuple{Float, Float, Float}, interpolator_enum=Linear_en::Interpolator_enum, keep_begining_same::Bool=false, extrapolate_value=0)::Array
    itp = prep_itp(input_array, input_array_spacing, interpolator_enum, extrapolate_value) # Prepare the interpolator with appropriate settings.
    res=collect(range(1,size(points_to_interpolate)[2]))
    res=map(el->interpolate_point(points_to_interpolate[:,el], itp, keep_begining_same, extrapolate_value), res) # Map interpolation function across all points.

    return res
end

"""
`prep_itp(input_array::Array, input_array_spacing::Array, interpolator_enum=Linear_en::Interpolator_enum, extrapolate_value=0)`

Prepares an interpolation object based on the given input array, spacing, and interpolation method. This function configures the interpolation strategy, including extrapolation settings, to be used in spatial data processing.

# Arguments
- `input_array`: The data array from which interpolation will be performed.
- `input_array_spacing`: Spacing adjustments for each dimension of `input_array`. You can get them by using the Your_MedImage.spacing method.
- `interpolator_enum`: The type of interpolation method to use (e.g., Nearest neighbor, Linear, B-spline).
- `extrapolate_value`: The value used for extrapolation, returned where interpolation is outside the bounds of `input_array`.


# Returns
- `Interpolation`: An interpolation object configured with the specified properties and ready to be used for actual data interpolation.
"""
function prep_itp(input_array::Array, input_array_spacing::Array, interpolator_enum=Linear_en::Interpolator_enum, extrapolate_value=0)
    old_size=size(input_array) # Retrieve the size of the input data array.
    if interpolator_enum == Nearest_neighbour_en
        itp = interpolate(input_array, BSpline(Constant()))
    elseif interpolator_enum == Linear_en
        itp = interpolate(input_array, BSpline(Linear()))
    elseif interpolator_enum == B_spline_en
        itp = interpolate(input_array, BSpline(Cubic(Line(OnGrid()))))
    end

    # Setup axis scaling based on the input array spacing.
    A_x1 = 1:input_array_spacing[1]:(old_size[1]+input_array_spacing[1]*old_size[1])
    A_x2 = 1:input_array_spacing[2]:(old_size[2]+input_array_spacing[2]*old_size[2])
    A_x3 = 1:input_array_spacing[3]:(old_size[3]+input_array_spacing[3]*old_size[3])
    
    itp=extrapolate(itp, extrapolate_value) # Configure extrapolation with the specified value.
    itp = scale(itp, A_x1, A_x2, A_x3)
    return itp
end


"""
`TransformIndexToPhysicalPoint_julia(index::Tuple{Int, Int, Int}, origin::Tuple{Float, Float, Float}, spacing::Tuple{Float, Float, Float})::Array{Float}

Convert a voxel index to a physical point using the origin and spacing of the image data.

# Arguments
- `index`: specifying the voxel coordinates (x, y, z).
- `origin`: the physical world coordinates of the image origin. You can get them by using the Your_MedImage.origin.
- `spacing`: the distance between adjacent voxels in each dimension. You can get them by using the Your_MedImage.spacing.

# Returns
- `Array{Float}`: The physical point in space corresponding to the voxel index.
"""
function TransformIndexToPhysicalPoint_julia(index::Tuple{Int,Int,Int}, origin::Tuple{Float,Float,Float}, spacing::Tuple{Float,Float,Float})::Array{Float}
    return collect(collect(origin) .+ ((collect(index)) .* collect(spacing)))
end


"""
Convert an array or already existing tuple into a tuple format.
"""
function ensure_tuple(arr)
    if arr isa Tuple
        return arr
    elseif arr isa AbstractArray
        return tuple(arr...)
    else
        error("Input must be a Tuple or an AbstractArray")
    end
end


"""
`extrapolate_corner_median(image::Union{MedImage, Array{Float, 3}})::Float`

Compute the median value from the corner voxels of a 3D image. This allows you to simulate empty space from the image.
\nSupports `MedImage` or 3D `Float` arrays.

# Arguments
- `image`: The image data, either a specialized `MedImage` type or a standard 3D `Float` array.

# Returns
- `Float`: The median value of the corner voxels of the image.
"""
function extrapolate_corner_median(image::Union{MedImage, Array{Float, 3}})::Float
    im = union_check(image)
    corners = [
        im[1, 1, 1],
        im[1, 1, end],
        im[1, end, 1],
        im[1, end, end],
        im[end, 1, 1],
        im[end, 1, end],
        im[end, end, 1],
        im[end, end, end]
    ]
    value_to_extrapolate = median(corners)
    return value_to_extrapolate
end


"""
`union_check_input(image::Union{MedImage, Array{Float, 3}})::Array{Float, 3}`

Verify or convert the input image type, ensuring it is suitable for processing.
\nSupports `MedImage` or 3D `Float` arrays.

# Arguments
- `image`: MedImage object or image data.

# Returns
- `Array{Float, 3}`: A copy of the image data.

# Errors
- Throws an error if `image` is neither a `MedImage` nor a `Array{Float, 3}`.
"""
function union_check_input(image::Union{MedImage, Array{Float, 3}})::Array{Float, 3}
    if image isa MedImage
        return copy(image.voxel_data)
    elseif image isa Array{Float, 3}
        return copy(image)
    else
        error("Invalid input type. Use MedImage or Array{Float, 3}.")
    end
end

"""
`union_check_output(original_image::Union{MedImage, Array{Float, 3}}, image::Array{Float, 3})::Union{MedImage, Array{Float, 3}}`

Convert or update the processed image to match the type of the original image input.

# Arguments
- `original_image`: The original image data.
- `image`: Processed image data as a 3D `Float` array.

# Returns
- Depending on the type of `original_image`, either an updated `MedImage` or the `Array{Float, 3}`.
"""
function union_check_output(original_image::Union{MedImage, Array{Float, 3}}, image::Array{Float, 3})::Union{MedImage, Array{Float, 3}}
    if original_image isa MedImage
        return update_voxel_data(original_image, image)
    elseif original_image isa Array{Float, 3}
        return image
    end
end


"""
`backend_check(processing_unit::String, backend)`

Adjust the backend data structure according to the specified processing unit to optimize computational performance.

# Arguments
- `processing_unit`: The type of hardware to be used for processing (`CPU`, `GPU`, or `AMD`).
- `backend`: The array to be adjusted.

# Returns
- The array adjusted for the specified backend.

# Errors
- Throws an error if `processing_unit` is not recognized available options are `CPU`, `GPU`, or `AMD`.
"""
function backend_check(processing_unit::String, backend::Array)
    if processing_unit == "CPU"
        return backend
    elseif processing_unit == "GPU"
        return CuArray(backend)
    elseif processing_unit == "AMD"
        return ROCarray(backend)
    else
        error("Invalid processing unit. Choose 'CPU', 'GPU', or 'AMD'.")
    end
end



