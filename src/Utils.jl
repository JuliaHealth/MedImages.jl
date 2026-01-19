module Utils
using ..MedImage_data_struct
using CUDA
using KernelAbstractions, Interpolations
import KernelAbstractions: synchronize, get_backend
using Enzyme
using ChainRulesCore
import Random

export interpolate_point
export get_base_indicies_arr
export cast_to_array_b_type
export interpolate_my
export TransformIndexToPhysicalPoint_julia
export ensure_tuple
export create_nii_from_medimage
export resample_kernel_launch
export is_cuda_array, extract_corners

import ..MedImage_data_struct: MedImage, Interpolator_enum, Mode_mi, Orientation_code, Nearest_neighbour_en, Linear_en, B_spline_en

# CUDA detection utilities
"""
Check if array is on GPU
"""
is_cuda_array(arr) = isa(arr, CuArray)

"""
Extract 8 corner values safely from CUDA or CPU arrays.
Uses @allowscalar for GPU arrays since 8 scalar reads is faster than kernel setup.
"""
function extract_corners(arr::AbstractArray{T,3}) where T
    dims = size(arr)
    if is_cuda_array(arr)
        # Allow 8 scalar reads - faster than kernel setup for so few values
        return CUDA.@allowscalar [
            arr[1, 1, 1], arr[1, 1, dims[3]],
            arr[1, dims[2], 1], arr[1, dims[2], dims[3]],
            arr[dims[1], 1, 1], arr[dims[1], 1, dims[3]],
            arr[dims[1], dims[2], 1], arr[dims[1], dims[2], dims[3]]
        ]
    else
        return [arr[1,1,1], arr[1,1,end], arr[1,end,1], arr[1,end,end],
                arr[end,1,1], arr[end,1,end], arr[end,end,1], arr[end,end,end]]
    end
end

"""
return array of cartesian indices for given dimensions in a form of array
Optimized: uses vectorized broadcasting instead of scalar loops
"""
function get_base_indicies_arr(dims)
    n_points = prod(dims)

    # Use broadcasting to generate indices efficiently
    # This is much faster than triple-nested loops for large arrays
    i_indices = repeat(1:dims[1], dims[2] * dims[3])
    j_indices = repeat(repeat(1:dims[2], inner=dims[1]), dims[3])
    k_indices = repeat(1:dims[3], inner=dims[1] * dims[2])

    # Stack into 3xN matrix
    indices = Matrix{Int32}(undef, 3, n_points)
    indices[1, :] = i_indices
    indices[2, :] = j_indices
    indices[3, :] = k_indices

    return indices
end#get_base_indicies_arr

# Mark as non-differentiable - indices are not differentiated
ChainRulesCore.@non_differentiable get_base_indicies_arr(::Any)

# Mark as non-differentiable - corner extraction is for computing extrapolation values
ChainRulesCore.@non_differentiable extract_corners(::Any)

"""
cast array a to the value type of array b
"""
function cast_to_array_b_type(a, b)
    # Check if array a and b have the same type
    if eltype(a) != eltype(b)
        # Cast array a to the value type of array b
        if eltype(b) in [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64]
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
function interpolate_point(point, itp, keep_begining_same=false, extrapolate_value=0)

    i = point[1]
    j = point[2]
    k = point[3]
    if (i < 0 || j < 0 || k < 0)
        return extrapolate_value
    end


    if (keep_begining_same)
        if ((i < 1))
            i = 1
        end
        if ((j < 1))
            j = 1
        end
        if ((k < 1))
            k = 1
        end
    end
    return itp(i, j, k)

end#interpolate_point




@kernel function interpolate_kernel(out_res, @Const(source_arr_shape), @Const(source_arr), @Const(points_to_interpolate), @Const(spacing), keep_begining_same, extrapolate_value, is_nearest_neighbour)
    shared_arr = @localmem(Float32, (512, 3))

    index_local = @index(Local, Linear)
    I = @index(Global)

    # Use shared memory to buffer points (coalesced read)
    shared_arr[index_local, 1] = points_to_interpolate[1, I]
    shared_arr[index_local, 2] = points_to_interpolate[2, I]
    shared_arr[index_local, 3] = points_to_interpolate[3, I]

    # Convert physical coordinates to index space
    real_x = (shared_arr[index_local, 1] - 1.0f0) / Float32(spacing[1]) + 1.0f0
    real_y = (shared_arr[index_local, 2] - 1.0f0) / Float32(spacing[2]) + 1.0f0
    real_z = (shared_arr[index_local, 3] - 1.0f0) / Float32(spacing[3]) + 1.0f0

    # Bounds check
    if real_x < 1.0f0 || real_y < 1.0f0 || real_z < 1.0f0 || real_x > Float32(source_arr_shape[1]) || real_y > Float32(source_arr_shape[2]) || real_z > Float32(source_arr_shape[3])
        out_res[I] = extrapolate_value
    else
        # Handle keep_beginning_same logic
        if keep_begining_same
            real_x = max(real_x, 1.0f0)
            real_y = max(real_y, 1.0f0)
            real_z = max(real_z, 1.0f0)
        end

        if is_nearest_neighbour
            # Nearest Neighbor
            out_res[I] = source_arr[Int(round(real_x)), Int(round(real_y)), Int(round(real_z))]
        else
            # Trilinear Interpolation (Optimized)
            x0 = floor(Int, real_x)
            y0 = floor(Int, real_y)
            z0 = floor(Int, real_z)

            x1 = min(x0 + 1, source_arr_shape[1])
            y1 = min(y0 + 1, source_arr_shape[2])
            z1 = min(z0 + 1, source_arr_shape[3])

            xd = real_x - x0
            yd = real_y - y0
            zd = real_z - z0

            @inbounds begin
                # Interpolate inline to minimize variables
                out_res[I] = (1.0f0 - zd) * (
                    (1.0f0 - yd) * (Float32(source_arr[x0, y0, z0]) * (1.0f0 - xd) + Float32(source_arr[x1, y0, z0]) * xd) +
                    yd * (Float32(source_arr[x0, y1, z0]) * (1.0f0 - xd) + Float32(source_arr[x1, y1, z0]) * xd)
                ) + zd * (
                    (1.0f0 - yd) * (Float32(source_arr[x0, y0, z1]) * (1.0f0 - xd) + Float32(source_arr[x1, y0, z1]) * xd) +
                    yd * (Float32(source_arr[x0, y1, z1]) * (1.0f0 - xd) + Float32(source_arr[x1, y1, z1]) * xd)
                )
            end
        end
    end
end

# Pure Julia CPU interpolation without KernelAbstractions (for Enzyme compatibility)
function interpolate_cpu_loop!(out_res, source_arr_shape, source_arr, points_to_interpolate, spacing, keep_begining_same, extrapolate_value, is_nearest_neighbour)
    n_points = size(points_to_interpolate, 2)
    @inbounds for I in 1:n_points
        # Convert physical coordinates to index space
        real_x = (points_to_interpolate[1, I] - 1.0f0) / Float32(spacing[1]) + 1.0f0
        real_y = (points_to_interpolate[2, I] - 1.0f0) / Float32(spacing[2]) + 1.0f0
        real_z = (points_to_interpolate[3, I] - 1.0f0) / Float32(spacing[3]) + 1.0f0

        # Bounds check
        if real_x < 1.0f0 || real_y < 1.0f0 || real_z < 1.0f0 || real_x > Float32(source_arr_shape[1]) || real_y > Float32(source_arr_shape[2]) || real_z > Float32(source_arr_shape[3])
            out_res[I] = extrapolate_value
        else
            # Handle keep_beginning_same logic
            if keep_begining_same
                real_x = max(real_x, 1.0f0)
                real_y = max(real_y, 1.0f0)
                real_z = max(real_z, 1.0f0)
            end

            if is_nearest_neighbour
                # Nearest Neighbor
                out_res[I] = source_arr[Int(round(real_x)), Int(round(real_y)), Int(round(real_z))]
            else
                # Trilinear Interpolation
                x0 = floor(Int, real_x)
                y0 = floor(Int, real_y)
                z0 = floor(Int, real_z)

                x1 = min(x0 + 1, source_arr_shape[1])
                y1 = min(y0 + 1, source_arr_shape[2])
                z1 = min(z0 + 1, source_arr_shape[3])

                xd = real_x - x0
                yd = real_y - y0
                zd = real_z - z0

                out_res[I] = (1.0f0 - zd) * (
                    (1.0f0 - yd) * (Float32(source_arr[x0, y0, z0]) * (1.0f0 - xd) + Float32(source_arr[x1, y0, z0]) * xd) +
                    yd * (Float32(source_arr[x0, y1, z0]) * (1.0f0 - xd) + Float32(source_arr[x1, y1, z0]) * xd)
                ) + zd * (
                    (1.0f0 - yd) * (Float32(source_arr[x0, y0, z1]) * (1.0f0 - xd) + Float32(source_arr[x1, y0, z1]) * xd) +
                    yd * (Float32(source_arr[x0, y1, z1]) * (1.0f0 - xd) + Float32(source_arr[x1, y1, z1]) * xd)
                )
            end
        end
    end
    return nothing
end

function interpolate_pure(points_to_interpolate, input_array, input_array_spacing, keep_begining_same, extrapolate_value, is_nearest_neighbour)
    out_res = similar(points_to_interpolate, eltype(points_to_interpolate), size(points_to_interpolate, 2))
    backend = get_backend(points_to_interpolate)
    source_arr_shape = size(input_array)

    if backend isa KernelAbstractions.CPU
        # Use pure Julia loop on CPU for better Enzyme compatibility
        interpolate_cpu_loop!(out_res, source_arr_shape, input_array, points_to_interpolate, input_array_spacing, keep_begining_same, extrapolate_value, is_nearest_neighbour)
    else
        # Use KA kernel on GPU
        interpolate_kernel(backend, 512)(out_res, source_arr_shape, input_array, points_to_interpolate, input_array_spacing, keep_begining_same, extrapolate_value, is_nearest_neighbour, ndrange=size(out_res))
        synchronize(backend)
    end
    return out_res
end

function ChainRulesCore.rrule(::typeof(interpolate_pure), points_to_interpolate, input_array, input_array_spacing, keep_begining_same, extrapolate_value, is_nearest_neighbour)
    output = interpolate_pure(points_to_interpolate, input_array, input_array_spacing, keep_begining_same, extrapolate_value, is_nearest_neighbour)

    function interpolate_pullback(d_output_unthunked)
        d_output_raw = unthunk(d_output_unthunked)
        d_points = zero(points_to_interpolate)
        d_input = zero(input_array)

        backend = get_backend(points_to_interpolate)
        source_arr_shape = size(input_array)

        if backend isa KernelAbstractions.CPU
            # Use pure Julia loop for CPU - Enzyme compatible
            d_output = d_output_raw
            function cpu_wrapper(out, src, pts, sp, kbs, ev, inn)
                interpolate_cpu_loop!(out, source_arr_shape, src, pts, sp, kbs, ev, inn)
                return nothing
            end

            Enzyme.autodiff(
                Reverse,
                cpu_wrapper,
                Const,
                Duplicated(output, d_output),
                Duplicated(input_array, d_input),
                Duplicated(points_to_interpolate, d_points),
                Const(input_array_spacing),
                Const(keep_begining_same),
                Const(extrapolate_value),
                Const(is_nearest_neighbour)
            )
        else
            # Use KA kernel for GPU - ensure gradient arrays are on the same device
            d_output = is_cuda_array(output) && !is_cuda_array(d_output_raw) ? CuArray(d_output_raw) : d_output_raw
            function kernel_wrapper(out, src, pts, sp, kbs, ev, inn)
                 interpolate_kernel(backend, 512)(out, source_arr_shape, src, pts, sp, kbs, ev, inn, ndrange=size(out))
                 return nothing
            end

            Enzyme.autodiff(
                Reverse,
                kernel_wrapper,
                Const,
                Duplicated(output, d_output),
                Duplicated(input_array, d_input),
                Duplicated(points_to_interpolate, d_points),
                Const(input_array_spacing),
                Const(keep_begining_same),
                Const(extrapolate_value),
                Const(is_nearest_neighbour)
            )
        end
        # Convert gradients back to CPU if needed for Zygote
        d_points_out = is_cuda_array(d_points) ? Array(d_points) : d_points
        d_input_out = is_cuda_array(d_input) ? Array(d_input) : d_input
        return NoTangent(), d_points_out, d_input_out, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    return output, interpolate_pullback
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
function interpolate_my(points_to_interpolate, input_array, input_array_spacing, interpolator_enum, keep_begining_same, extrapolate_value=0, use_fast=true)

    old_size = size(input_array)
    interpolator_enum = Int(interpolator_enum)
    interpolator_enum = instances(Interpolator_enum)[interpolator_enum+1]

    if (use_fast)
        is_nearest_neighbour = (interpolator_enum == Nearest_neighbour_en)

        # Ensure points_to_interpolate is on the same backend as input_array
        # This is necessary for CUDA arrays to work correctly
        if is_cuda_array(input_array) && !is_cuda_array(points_to_interpolate)
            points_to_interpolate = CuArray(Float32.(points_to_interpolate))
        end

        # Call pure function for differentiability (has rrule defined)
        return interpolate_pure(points_to_interpolate, input_array, input_array_spacing, keep_begining_same, extrapolate_value, is_nearest_neighbour)
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

    A_x1 = 1:input_array_spacing[1]:(1+input_array_spacing[1]*(old_size[1]-1))
    A_x2 = 1:input_array_spacing[2]:(1+input_array_spacing[2]*(old_size[2]-1))
    A_x3 = 1:input_array_spacing[3]:(1+input_array_spacing[3]*(old_size[3]-1))


    itp = extrapolate(itp, extrapolate_value)
    itp = scale(itp, A_x1, A_x2, A_x3)
    # Create the new voxel data
    # print("eeeeeeeeeeeeee $(itp(-1222.0,-1222.0,-1222.0))")


    res = collect(range(1, size(points_to_interpolate)[2]))
    res = map(el -> interpolate_point(points_to_interpolate[:, el], itp, keep_begining_same, extrapolate_value), res)

    return res
end#interpolate_my



function TransformIndexToPhysicalPoint_julia(index::Tuple{Int,Int,Int}, origin::Tuple{Float64,Float64,Float64}, spacing::Tuple{Float64,Float64,Float64})

    # return origin .+ ((collect(index) .- 1) .* collect(spacing))
    return collect(collect(origin) .+ ((collect(index)) .* collect(spacing)))
end

function ensure_tuple(arr)
    if isa(arr, Tuple)
        return arr
    elseif isa(arr, AbstractArray)
        return Tuple(arr)
    else
        error("Cannot convert to tuple: $arr")
    end
end

@kernel function trilinear_resample_kernel(output, @Const(image_data), @Const(old_spacing), @Const(new_spacing), @Const(new_dims))
    i = @index(Global, Linear)

    # Map linear index to x,y,z (1-based)
    stride_z = new_dims[1] * new_dims[2]
    iz = (i - 1) ÷ stride_z + 1
    rem_z = (i - 1) % stride_z
    iy = rem_z ÷ new_dims[1] + 1
    ix = rem_z % new_dims[1] + 1

    # Map to source index space
    real_x = (Float32(ix) - 1.0f0) * (Float32(new_spacing[1]) / Float32(old_spacing[1])) + 1.0f0
    real_y = (Float32(iy) - 1.0f0) * (Float32(new_spacing[2]) / Float32(old_spacing[2])) + 1.0f0
    real_z = (Float32(iz) - 1.0f0) * (Float32(new_spacing[3]) / Float32(old_spacing[3])) + 1.0f0

    # Bounds checking
    sx, sy, sz = size(image_data)
    if real_x < 1.0f0 || real_y < 1.0f0 || real_z < 1.0f0 || real_x > Float32(sx) || real_y > Float32(sy) || real_z > Float32(sz)
        output[i] = 0 # Extrapolation value 0
    else
        x0 = floor(Int, real_x)
        y0 = floor(Int, real_y)
        z0 = floor(Int, real_z)

        # Clamp upper bounds
        x1 = min(x0 + 1, sx)
        y1 = min(y0 + 1, sy)
        z1 = min(z0 + 1, sz)

        xd = real_x - x0
        yd = real_y - y0
        zd = real_z - z0

        # Inline Interpolation with @inbounds
        @inbounds begin
            c00 = Float32(image_data[x0, y0, z0]) * (1.0f0 - xd) + Float32(image_data[x1, y0, z0]) * xd
            c10 = Float32(image_data[x0, y1, z0]) * (1.0f0 - xd) + Float32(image_data[x1, y1, z0]) * xd
            c01 = Float32(image_data[x0, y0, z1]) * (1.0f0 - xd) + Float32(image_data[x1, y0, z1]) * xd
            c11 = Float32(image_data[x0, y1, z1]) * (1.0f0 - xd) + Float32(image_data[x1, y1, z1]) * xd

            c0 = c00 * (1.0f0 - yd) + c10 * yd
            c1 = c01 * (1.0f0 - yd) + c11 * yd

            output[i] = c0 * (1.0f0 - zd) + c1 * zd
        end
    end
end

@kernel function nearest_resample_kernel(output, @Const(image_data), @Const(old_spacing), @Const(new_spacing), @Const(new_dims))
    i = @index(Global, Linear)

    iz = (i - 1) ÷ (new_dims[1] * new_dims[2]) + 1
    rem_z = (i - 1) % (new_dims[1] * new_dims[2])
    iy = rem_z ÷ new_dims[1] + 1
    ix = rem_z % new_dims[1] + 1

    real_x = (Float32(ix) - 1.0f0) * (Float32(new_spacing[1]) / Float32(old_spacing[1])) + 1.0f0
    real_y = (Float32(iy) - 1.0f0) * (Float32(new_spacing[2]) / Float32(old_spacing[2])) + 1.0f0
    real_z = (Float32(iz) - 1.0f0) * (Float32(new_spacing[3]) / Float32(old_spacing[3])) + 1.0f0

    src_size = size(image_data)

    # Nearest neighbor rounding
    nx = Int(round(real_x))
    ny = Int(round(real_y))
    nz = Int(round(real_z))

    if nx < 1 || ny < 1 || nz < 1 || nx > src_size[1] || ny > src_size[2] || nz > src_size[3]
        output[i] = 0
    else
        output[i] = image_data[nx, ny, nz]
    end
end

# =============================================================================
# Enzyme-compatible kernels for GPU autodiff
# These use KernelAbstractions without @Const annotations, matching the pattern
# that works with Enzyme for automatic differentiation on GPU
# =============================================================================

@kernel function trilinear_resample_enzyme_kernel!(output, image_data, old_spacing_arr, new_spacing_arr, new_dims_arr, src_dims_arr)
    i = @index(Global, Linear)

    # Read dims from arrays (not tuples)
    ndx = Int(new_dims_arr[1])
    ndy = Int(new_dims_arr[2])

    # Map linear index to x,y,z (1-based)
    stride_z = ndx * ndy
    iz = (i - 1) ÷ stride_z + 1
    rem_z = (i - 1) % stride_z
    iy = rem_z ÷ ndx + 1
    ix = rem_z % ndx + 1

    # Map to source index space using array indexing
    real_x = (Float32(ix) - 1.0f0) * (Float32(new_spacing_arr[1]) / Float32(old_spacing_arr[1])) + 1.0f0
    real_y = (Float32(iy) - 1.0f0) * (Float32(new_spacing_arr[2]) / Float32(old_spacing_arr[2])) + 1.0f0
    real_z = (Float32(iz) - 1.0f0) * (Float32(new_spacing_arr[3]) / Float32(old_spacing_arr[3])) + 1.0f0

    # Bounds checking - use passed dims instead of size()
    sx = Int(src_dims_arr[1])
    sy = Int(src_dims_arr[2])
    sz = Int(src_dims_arr[3])

    if real_x < 1.0f0 || real_y < 1.0f0 || real_z < 1.0f0 || real_x > Float32(sx) || real_y > Float32(sy) || real_z > Float32(sz)
        output[i] = 0.0f0
    else
        x0 = floor(Int, real_x)
        y0 = floor(Int, real_y)
        z0 = floor(Int, real_z)

        # Clamp upper bounds
        x1 = min(x0 + 1, sx)
        y1 = min(y0 + 1, sy)
        z1 = min(z0 + 1, sz)

        xd = real_x - x0
        yd = real_y - y0
        zd = real_z - z0

        # Trilinear interpolation
        @inbounds begin
            c00 = Float32(image_data[x0, y0, z0]) * (1.0f0 - xd) + Float32(image_data[x1, y0, z0]) * xd
            c10 = Float32(image_data[x0, y1, z0]) * (1.0f0 - xd) + Float32(image_data[x1, y1, z0]) * xd
            c01 = Float32(image_data[x0, y0, z1]) * (1.0f0 - xd) + Float32(image_data[x1, y0, z1]) * xd
            c11 = Float32(image_data[x0, y1, z1]) * (1.0f0 - xd) + Float32(image_data[x1, y1, z1]) * xd

            c0 = c00 * (1.0f0 - yd) + c10 * yd
            c1 = c01 * (1.0f0 - yd) + c11 * yd

            output[i] = c0 * (1.0f0 - zd) + c1 * zd
        end
    end
end

@kernel function nearest_resample_enzyme_kernel!(output, image_data, old_spacing_arr, new_spacing_arr, new_dims_arr, src_dims_arr)
    i = @index(Global, Linear)

    # Read dims from arrays (not tuples)
    ndx = Int(new_dims_arr[1])
    ndy = Int(new_dims_arr[2])

    iz = (i - 1) ÷ (ndx * ndy) + 1
    rem_z = (i - 1) % (ndx * ndy)
    iy = rem_z ÷ ndx + 1
    ix = rem_z % ndx + 1

    real_x = (Float32(ix) - 1.0f0) * (Float32(new_spacing_arr[1]) / Float32(old_spacing_arr[1])) + 1.0f0
    real_y = (Float32(iy) - 1.0f0) * (Float32(new_spacing_arr[2]) / Float32(old_spacing_arr[2])) + 1.0f0
    real_z = (Float32(iz) - 1.0f0) * (Float32(new_spacing_arr[3]) / Float32(old_spacing_arr[3])) + 1.0f0

    # Use passed dims instead of size()
    sx = Int(src_dims_arr[1])
    sy = Int(src_dims_arr[2])
    sz = Int(src_dims_arr[3])

    # Nearest neighbor rounding
    nx = Int(round(real_x))
    ny = Int(round(real_y))
    nz = Int(round(real_z))

    if nx < 1 || ny < 1 || nz < 1 || nx > sx || ny > sy || nz > sz
        output[i] = 0.0f0
    else
        @inbounds output[i] = Float32(image_data[nx, ny, nz])
    end
end

# Launcher functions for Enzyme-compatible kernels
# These convert tuples to arrays and match the working example pattern
# NOTE: ndrange_val is passed as a scalar to avoid scalar indexing on GPU arrays

function trilinear_enzyme_launcher!(out, img, osp_arr, nsp_arr, ndims_arr, src_dims_arr, ndrange_val)
    backend = KernelAbstractions.get_backend(out)
    kernel = trilinear_resample_enzyme_kernel!(backend, 256)
    kernel(out, img, osp_arr, nsp_arr, ndims_arr, src_dims_arr, ndrange=ndrange_val)
    KernelAbstractions.synchronize(backend)
    return nothing
end

function nearest_enzyme_launcher!(out, img, osp_arr, nsp_arr, ndims_arr, src_dims_arr, ndrange_val)
    backend = KernelAbstractions.get_backend(out)
    kernel = nearest_resample_enzyme_kernel!(backend, 256)
    kernel(out, img, osp_arr, nsp_arr, ndims_arr, src_dims_arr, ndrange=ndrange_val)
    KernelAbstractions.synchronize(backend)
    return nothing
end

# CPU version of trilinear resample loop (Enzyme compatible)
function trilinear_resample_cpu_loop!(output, image_data, old_spacing, new_spacing, new_dims)
    n_points = prod(new_dims)
    stride_z = new_dims[1] * new_dims[2]
    sx, sy, sz = size(image_data)

    @inbounds for i in 1:n_points
        iz = (i - 1) ÷ stride_z + 1
        rem_z = (i - 1) % stride_z
        iy = rem_z ÷ new_dims[1] + 1
        ix = rem_z % new_dims[1] + 1

        real_x = (Float32(ix) - 1.0f0) * (Float32(new_spacing[1]) / Float32(old_spacing[1])) + 1.0f0
        real_y = (Float32(iy) - 1.0f0) * (Float32(new_spacing[2]) / Float32(old_spacing[2])) + 1.0f0
        real_z = (Float32(iz) - 1.0f0) * (Float32(new_spacing[3]) / Float32(old_spacing[3])) + 1.0f0

        if real_x < 1.0f0 || real_y < 1.0f0 || real_z < 1.0f0 || real_x > Float32(sx) || real_y > Float32(sy) || real_z > Float32(sz)
            output[i] = 0
        else
            x0 = floor(Int, real_x)
            y0 = floor(Int, real_y)
            z0 = floor(Int, real_z)

            x1 = min(x0 + 1, sx)
            y1 = min(y0 + 1, sy)
            z1 = min(z0 + 1, sz)

            xd = real_x - x0
            yd = real_y - y0
            zd = real_z - z0

            c00 = Float32(image_data[x0, y0, z0]) * (1.0f0 - xd) + Float32(image_data[x1, y0, z0]) * xd
            c10 = Float32(image_data[x0, y1, z0]) * (1.0f0 - xd) + Float32(image_data[x1, y1, z0]) * xd
            c01 = Float32(image_data[x0, y0, z1]) * (1.0f0 - xd) + Float32(image_data[x1, y0, z1]) * xd
            c11 = Float32(image_data[x0, y1, z1]) * (1.0f0 - xd) + Float32(image_data[x1, y1, z1]) * xd

            c0 = c00 * (1.0f0 - yd) + c10 * yd
            c1 = c01 * (1.0f0 - yd) + c11 * yd

            output[i] = c0 * (1.0f0 - zd) + c1 * zd
        end
    end
    return nothing
end

# CPU version of nearest resample loop (Enzyme compatible)
function nearest_resample_cpu_loop!(output, image_data, old_spacing, new_spacing, new_dims)
    n_points = prod(new_dims)
    stride_z = new_dims[1] * new_dims[2]
    sx, sy, sz = size(image_data)

    @inbounds for i in 1:n_points
        iz = (i - 1) ÷ stride_z + 1
        rem_z = (i - 1) % stride_z
        iy = rem_z ÷ new_dims[1] + 1
        ix = rem_z % new_dims[1] + 1

        real_x = (Float32(ix) - 1.0f0) * (Float32(new_spacing[1]) / Float32(old_spacing[1])) + 1.0f0
        real_y = (Float32(iy) - 1.0f0) * (Float32(new_spacing[2]) / Float32(old_spacing[2])) + 1.0f0
        real_z = (Float32(iz) - 1.0f0) * (Float32(new_spacing[3]) / Float32(old_spacing[3])) + 1.0f0

        nx = Int(round(real_x))
        ny = Int(round(real_y))
        nz = Int(round(real_z))

        if nx < 1 || ny < 1 || nz < 1 || nx > sx || ny > sy || nz > sz
            output[i] = 0
        else
            output[i] = image_data[nx, ny, nz]
        end
    end
    return nothing
end

function resample_kernel_launch(image_data, old_spacing, new_spacing, new_dims, interpolator_enum)
    # Output array
    output = similar(image_data, new_dims)

    # Select backend
    backend = get_backend(output)

    if backend isa KernelAbstractions.CPU
        # Use pure Julia loops on CPU for Enzyme compatibility
        if interpolator_enum == Nearest_neighbour_en
            nearest_resample_cpu_loop!(vec(output), image_data, old_spacing, new_spacing, new_dims)
        else
            trilinear_resample_cpu_loop!(vec(output), image_data, old_spacing, new_spacing, new_dims)
        end
    else
        # Use KA kernels on GPU
        if interpolator_enum == Nearest_neighbour_en
            kernel = nearest_resample_kernel(backend, 256)
        else
            kernel = trilinear_resample_kernel(backend, 256)
        end
        kernel(output, image_data, old_spacing, new_spacing, new_dims, ndrange=prod(new_dims))
        synchronize(backend)
    end

    return output
end

function ChainRulesCore.rrule(::typeof(resample_kernel_launch), image_data, old_spacing, new_spacing, new_dims, interpolator_enum)
    output = resample_kernel_launch(image_data, old_spacing, new_spacing, new_dims, interpolator_enum)

    function resample_pullback(d_output_unthunked)
        d_output_raw = unthunk(d_output_unthunked)
        d_image = zero(image_data)

        backend = get_backend(output)

        if backend isa KernelAbstractions.CPU
            # Use pure Julia loops for CPU - Enzyme compatible
            d_output = d_output_raw
            if interpolator_enum == Nearest_neighbour_en
                function nearest_cpu_wrapper(out, img, osp, nsp, ndims)
                    nearest_resample_cpu_loop!(out, img, osp, nsp, ndims)
                    return nothing
                end
                Enzyme.autodiff(
                    Reverse,
                    nearest_cpu_wrapper,
                    Const,
                    Duplicated(vec(output), vec(d_output)),
                    Duplicated(image_data, d_image),
                    Const(old_spacing),
                    Const(new_spacing),
                    Const(new_dims)
                )
            else
                function trilinear_cpu_wrapper(out, img, osp, nsp, ndims)
                    trilinear_resample_cpu_loop!(out, img, osp, nsp, ndims)
                    return nothing
                end
                Enzyme.autodiff(
                    Reverse,
                    trilinear_cpu_wrapper,
                    Const,
                    Duplicated(vec(output), vec(d_output)),
                    Duplicated(image_data, d_image),
                    Const(old_spacing),
                    Const(new_spacing),
                    Const(new_dims)
                )
            end
        else
            # GPU backward pass using Enzyme with new pattern (no @Const, arrays instead of tuples)
            # This matches the working example pattern for Enzyme + KernelAbstractions GPU autodiff

            # Ensure gradient arrays are on GPU
            d_output = is_cuda_array(output) && !is_cuda_array(d_output_raw) ? CuArray(d_output_raw) : d_output_raw
            d_image = is_cuda_array(image_data) && !is_cuda_array(d_image) ? CuArray(d_image) : d_image

            # Convert tuples to GPU arrays (matching working example pattern)
            old_spacing_arr = CuArray(Float32[old_spacing...])
            new_spacing_arr = CuArray(Float32[new_spacing...])
            new_dims_arr = CuArray(Int32[new_dims...])
            src_dims = size(image_data)
            src_dims_arr = CuArray(Int32[src_dims...])

            # Shadow arrays for spacing/dims (not differentiated but need Duplicated for pattern)
            d_osp = CUDA.zeros(Float32, 3)
            d_nsp = CUDA.zeros(Float32, 3)
            d_ndims = CUDA.zeros(Int32, 3)
            d_src_dims = CUDA.zeros(Int32, 3)

            # ndrange computed as scalar to avoid GPU scalar indexing in launcher
            ndrange_val = prod(new_dims)

            if interpolator_enum == Nearest_neighbour_en
                Enzyme.autodiff(
                    Reverse,
                    nearest_enzyme_launcher!,
                    Duplicated(vec(output), vec(d_output)),
                    Duplicated(image_data, d_image),
                    Duplicated(old_spacing_arr, d_osp),
                    Duplicated(new_spacing_arr, d_nsp),
                    Duplicated(new_dims_arr, d_ndims),
                    Duplicated(src_dims_arr, d_src_dims),
                    Const(ndrange_val)
                )
            else
                Enzyme.autodiff(
                    Reverse,
                    trilinear_enzyme_launcher!,
                    Duplicated(vec(output), vec(d_output)),
                    Duplicated(image_data, d_image),
                    Duplicated(old_spacing_arr, d_osp),
                    Duplicated(new_spacing_arr, d_nsp),
                    Duplicated(new_dims_arr, d_ndims),
                    Duplicated(src_dims_arr, d_src_dims),
                    Const(ndrange_val)
                )
            end
        end

        # Convert gradients back to CPU if needed for Zygote
        d_image_out = is_cuda_array(d_image) ? Array(d_image) : d_image
        return NoTangent(), d_image_out, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    return output, resample_pullback
end

end#Utils
