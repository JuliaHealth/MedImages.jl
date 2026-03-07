module Utils
using ..MedImage_data_struct
using CUDA
using KernelAbstractions, Interpolations
import KernelAbstractions: synchronize, get_backend
using Enzyme
using ChainRulesCore
import Random
using LinearAlgebra

export interpolate_point
export get_base_indicies_arr
export cast_to_array_b_type
export interpolate_my, interpolate_fused_affine
export TransformIndexToPhysicalPoint_julia
export ensure_tuple
export create_nii_from_medimage
export resample_kernel_launch
export is_cuda_array, extract_corners
export create_batched_medimage, unbatch_medimage
export interpolate_fused_affine
export generate_affine_coords

import ..MedImage_data_struct: MedImage, BatchedMedImage, Interpolator_enum, Mode_mi, Orientation_code, Nearest_neighbour_en, Linear_en, B_spline_en

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
    I = @index(Global)

    # Convert physical coordinates to index space
    real_x = (points_to_interpolate[1, I] - 1.0f0) / Float32(spacing[1]) + 1.0f0
    real_y = (points_to_interpolate[2, I] - 1.0f0) / Float32(spacing[2]) + 1.0f0
    real_z = (points_to_interpolate[3, I] - 1.0f0) / Float32(spacing[3]) + 1.0f0

    # Bounds check
    if real_x < 1.0f0 || real_y < 1.0f0 || real_z < 1.0f0 || real_x > Float32(source_arr_shape[1]) || real_y > Float32(source_arr_shape[2]) || real_z > Float32(source_arr_shape[3])
        out_res[I] = extrapolate_value[1]
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
                # Interpolate inline
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

@kernel function interpolate_kernel_4d(out_res, @Const(source_arr_shape), @Const(source_arr), @Const(points_to_interpolate), @Const(spacing_arr), keep_begining_same, extrapolate_value, is_nearest_neighbour, @Const(points_batch_stride))
    I = @index(Global)
    n_points = size(out_res, 1)
    idx_point = (I - 1) % n_points + 1
    idx_batch = (I - 1) ÷ n_points + 1

    px = points_to_interpolate[1, idx_point, 1 + (idx_batch - 1) * points_batch_stride]
    py = points_to_interpolate[2, idx_point, 1 + (idx_batch - 1) * points_batch_stride]
    pz = points_to_interpolate[3, idx_point, 1 + (idx_batch - 1) * points_batch_stride]

    sx = spacing_arr[1, idx_batch]
    sy = spacing_arr[2, idx_batch]
    sz = spacing_arr[3, idx_batch]

    real_x = (px - 1.0f0) / Float32(sx) + 1.0f0
    real_y = (py - 1.0f0) / Float32(sy) + 1.0f0
    real_z = (pz - 1.0f0) / Float32(sz) + 1.0f0

    if real_x < 1.0f0 || real_y < 1.0f0 || real_z < 1.0f0 || real_x > Float32(source_arr_shape[1]) || real_y > Float32(source_arr_shape[2]) || real_z > Float32(source_arr_shape[3])
        out_res[I] = extrapolate_value[1]
    else
        if keep_begining_same
            real_x = max(real_x, 1.0f0)
            real_y = max(real_y, 1.0f0)
            real_z = max(real_z, 1.0f0)
        end

        if is_nearest_neighbour
            out_res[I] = source_arr[Int(round(real_x)), Int(round(real_y)), Int(round(real_z)), idx_batch]
        else
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
                out_res[I] = (1.0f0 - zd) * (
                    (1.0f0 - yd) * (Float32(source_arr[x0, y0, z0, idx_batch]) * (1.0f0 - xd) + Float32(source_arr[x1, y0, z0, idx_batch]) * xd) +
                    yd * (Float32(source_arr[x0, y1, z0, idx_batch]) * (1.0f0 - xd) + Float32(source_arr[x1, y1, z0, idx_batch]) * xd)
                ) + zd * (
                    (1.0f0 - yd) * (Float32(source_arr[x0, y0, z1, idx_batch]) * (1.0f0 - xd) + Float32(source_arr[x1, y0, z1, idx_batch]) * xd) +
                    yd * (Float32(source_arr[x0, y1, z1, idx_batch]) * (1.0f0 - xd) + Float32(source_arr[x1, y1, z1, idx_batch]) * xd)
                )
            end
        end
    end
end


@inline function fused_affine_point_logic(source_arr, affine_matrices, source_arr_shape, out_size_ka, center_shift, keep_begining_same, extrapolate_value, is_nearest_neighbour, ix, iy, iz, idx_batch)
    # Shift to center
    px = Float32(ix) - center_shift[1]
    py = Float32(iy) - center_shift[2]
    pz = Float32(iz) - center_shift[3]
    
    # Which matrix to use?
    mat_idx = size(affine_matrices, 3) == 1 ? 1 : idx_batch
    
    # Apply inverse affine matrix: p_source = M_inv * p_output
    new_px = affine_matrices[1,1,mat_idx]*px + affine_matrices[1,2,mat_idx]*py + affine_matrices[1,3,mat_idx]*pz + affine_matrices[1,4,mat_idx]
    new_py = affine_matrices[2,1,mat_idx]*px + affine_matrices[2,2,mat_idx]*py + affine_matrices[2,3,mat_idx]*pz + affine_matrices[2,4,mat_idx]
    new_pz = affine_matrices[3,1,mat_idx]*px + affine_matrices[3,2,mat_idx]*py + affine_matrices[3,3,mat_idx]*pz + affine_matrices[3,4,mat_idx]
    
    # Shift back to find real index in source
    real_x = new_px + center_shift[1]
    real_y = new_py + center_shift[2]
    real_z = new_pz + center_shift[3]
    
    # Bounds check
    if real_x < 1.0f0 || real_y < 1.0f0 || real_z < 1.0f0 || real_x > Float32(source_arr_shape[1]) || real_y > Float32(source_arr_shape[2]) || real_z > Float32(source_arr_shape[3])
        return extrapolate_value
    else
        # Handle keep_beginning_same logic
        if keep_begining_same
            real_x = max(real_x, 1.0f0)
            real_y = max(real_y, 1.0f0)
            real_z = max(real_z, 1.0f0)
        end

        if is_nearest_neighbour
            # Nearest Neighbor
            return Float32(source_arr[Int(round(real_x)), Int(round(real_y)), Int(round(real_z)), idx_batch])
        else
            # Trilinear Interpolation
            x0 = floor(Int, real_x)
            y0 = floor(Int, real_y)
            z0 = floor(Int, real_z)

            x1 = min(x0 + 1, Int(source_arr_shape[1]))
            y1 = min(y0 + 1, Int(source_arr_shape[2]))
            z1 = min(z0 + 1, Int(source_arr_shape[3]))

            xd = real_x - x0
            yd = real_y - y0
            zd = real_z - z0

            @inbounds begin
                c00 = Float32(source_arr[x0, y0, z0, idx_batch]) * (1.0f0 - xd) + Float32(source_arr[x1, y0, z0, idx_batch]) * xd
                c10 = Float32(source_arr[x0, y1, z0, idx_batch]) * (1.0f0 - xd) + Float32(source_arr[x1, y1, z0, idx_batch]) * xd
                c01 = Float32(source_arr[x0, y0, z1, idx_batch]) * (1.0f0 - xd) + Float32(source_arr[x1, y0, z1, idx_batch]) * xd
                c11 = Float32(source_arr[x0, y1, z1, idx_batch]) * (1.0f0 - xd) + Float32(source_arr[x1, y1, z1, idx_batch]) * xd

                c0 = c00 * (1.0f0 - yd) + c10 * yd
                c1 = c01 * (1.0f0 - yd) + c11 * yd

                return c0 * (1.0f0 - zd) + c1 * zd
            end
        end
    end
end

@kernel function fused_affine_interpolate_kernel(out_res, @Const(source_arr_shape), @Const(source_arr), @Const(affine_matrices), @Const(output_size), @Const(center_shift), keep_begining_same, extrapolate_value, is_nearest_neighbour)
    I = @index(Global)
    
    n_spatial = output_size[1] * output_size[2] * output_size[3]
    idx_spatial = (I - 1) % n_spatial + 1
    idx_batch = (I - 1) ÷ n_spatial + 1
    
    stride_z = output_size[1] * output_size[2]
    iz = (idx_spatial - 1) ÷ stride_z + 1
    rem_z = (idx_spatial - 1) % stride_z
    iy = rem_z ÷ output_size[1] + 1
    ix = rem_z % output_size[1] + 1
    
    out_res[I] = fused_affine_point_logic(source_arr, affine_matrices, source_arr_shape, output_size, center_shift, keep_begining_same, extrapolate_value[1], is_nearest_neighbour, ix, iy, iz, idx_batch)
end


function interpolate_cpu_loop_4d!(out_res, source_arr_shape, source_arr, points_to_interpolate, spacing_arr, keep_begining_same, extrapolate_value, is_nearest_neighbour, points_batch_stride)
    n_points = size(out_res, 1)
    batch_size = size(out_res, 2)

    @inbounds for b in 1:batch_size
        sx = spacing_arr[1, b]
        sy = spacing_arr[2, b]
        sz = spacing_arr[3, b]

        point_offset_idx = 1 + (b - 1) * points_batch_stride

        for i in 1:n_points
            # Read point
            px = points_to_interpolate[1, i, point_offset_idx]
            py = points_to_interpolate[2, i, point_offset_idx]
            pz = points_to_interpolate[3, i, point_offset_idx]

            # Convert physical coordinates to index space
            real_x = (px - 1.0f0) / Float32(sx) + 1.0f0
            real_y = (py - 1.0f0) / Float32(sy) + 1.0f0
            real_z = (pz - 1.0f0) / Float32(sz) + 1.0f0

            # Bounds check
            if real_x < 1.0f0 || real_y < 1.0f0 || real_z < 1.0f0 || real_x > Float32(source_arr_shape[1]) || real_y > Float32(source_arr_shape[2]) || real_z > Float32(source_arr_shape[3])
                out_res[i, b] = extrapolate_value
            else
                # Handle keep_beginning_same logic
                if keep_begining_same
                    real_x = max(real_x, 1.0f0)
                    real_y = max(real_y, 1.0f0)
                    real_z = max(real_z, 1.0f0)
                end

                if is_nearest_neighbour
                    # Nearest Neighbor
                    out_res[i, b] = source_arr[Int(round(real_x)), Int(round(real_y)), Int(round(real_z)), b]
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

                    out_res[i, b] = (1.0f0 - zd) * (
                        (1.0f0 - yd) * (Float32(source_arr[x0, y0, z0, b]) * (1.0f0 - xd) + Float32(source_arr[x1, y0, z0, b]) * xd) +
                        yd * (Float32(source_arr[x0, y1, z0, b]) * (1.0f0 - xd) + Float32(source_arr[x1, y1, z0, b]) * xd)
                    ) + zd * (
                        (1.0f0 - yd) * (Float32(source_arr[x0, y0, z1, b]) * (1.0f0 - xd) + Float32(source_arr[x1, y0, z1, b]) * xd) +
                        yd * (Float32(source_arr[x0, y1, z1, b]) * (1.0f0 - xd) + Float32(source_arr[x1, y1, z1, b]) * xd)
                    )
                end
            end
        end
    end
    return nothing
end

function interpolate_pure(points_to_interpolate, input_array, input_array_spacing, keep_begining_same, extrapolate_value, is_nearest_neighbour)
    if ndims(input_array) == 4
        # Batched mode
        n_points = size(points_to_interpolate, 2)
        batch_size = size(input_array, 4)

        # Prepare output
        out_res = similar(points_to_interpolate, Float32, (n_points, batch_size))
        backend = get_backend(points_to_interpolate)
        source_arr_shape = size(input_array)

        # Determine stride for points
        # points_to_interpolate can be (3, N) or (3, N, B)
        # If (3, N), we reshape to (3, N, 1) to unify kernel logic and set stride 0

        points_batch_stride = 0
        if ndims(points_to_interpolate) == 3 && size(points_to_interpolate, 3) == batch_size
             points_batch_stride = 1
        elseif ndims(points_to_interpolate) == 2
             points_batch_stride = 0
             # Reshape to 3D for kernel indexing if needed?
             # My kernel logic assumes it can index [1, i, 1].
             # On CPU normal arrays support this if we reshape.
             # On GPU, KA allows it if we view.
             # Actually, simpler to just reshape in place or view.
             points_to_interpolate = reshape(points_to_interpolate, size(points_to_interpolate, 1), size(points_to_interpolate, 2), 1)
        end

        # Spacing: assumed to be passed as Matrix (3, Batch) or Vector of Tuples
        # We need it as Matrix (3, Batch) for kernel
        if isa(input_array_spacing, Vector)
            # Convert to matrix
            spacing_mat = Matrix{Float32}(undef, 3, batch_size)
            for b in 1:batch_size
                spacing_mat[1, b] = Float32(input_array_spacing[b][1])
                spacing_mat[2, b] = Float32(input_array_spacing[b][2])
                spacing_mat[3, b] = Float32(input_array_spacing[b][3])
            end

            # Move to device if needed
            if backend isa KernelAbstractions.GPU
                 spacing_mat = CuArray(spacing_mat)
            end
            input_array_spacing = spacing_mat
        end

        # Prepare safe Const arguments for kernels (consistency with AD)
        extrapolate_value_arr = backend isa KernelAbstractions.GPU ? CuArray([Float32(extrapolate_value)]) : [Float32(extrapolate_value)]

        if backend isa KernelAbstractions.CPU
            # Use KA kernel on CPU
            interpolate_kernel_4d(backend, 512)(out_res, source_arr_shape, input_array, points_to_interpolate, input_array_spacing, keep_begining_same, extrapolate_value_arr, is_nearest_neighbour, points_batch_stride, ndrange=length(out_res))
            synchronize(backend)
        else
            # GPU
             interpolate_kernel_4d(backend, 512)(out_res, source_arr_shape, input_array, points_to_interpolate, input_array_spacing, keep_begining_same, extrapolate_value_arr, is_nearest_neighbour, points_batch_stride, ndrange=length(out_res))
             synchronize(backend)
        end
        return out_res
    else
        # Original 3D mode
        out_res = similar(points_to_interpolate, eltype(points_to_interpolate), size(points_to_interpolate, 2))
        backend = get_backend(points_to_interpolate)
        source_arr_shape = size(input_array)

        # Prepare safe Const arguments for kernels
        extrapolate_value_arr = backend isa KernelAbstractions.GPU ? CuArray([Float32(extrapolate_value)]) : [Float32(extrapolate_value)]
        spacing_arr = backend isa KernelAbstractions.GPU ? CuArray(Float32[input_array_spacing...]) : Float32[input_array_spacing...]

        if backend isa KernelAbstractions.CPU
            # Use KA kernel on CPU
            interpolate_kernel(backend, 512)(out_res, source_arr_shape, input_array, points_to_interpolate, spacing_arr, keep_begining_same, extrapolate_value_arr, is_nearest_neighbour, ndrange=size(out_res))
            synchronize(backend)
        else
            # Use KA kernel on GPU
            interpolate_kernel(backend, 512)(out_res, source_arr_shape, input_array, points_to_interpolate, spacing_arr, keep_begining_same, extrapolate_value_arr, is_nearest_neighbour, ndrange=size(out_res))
            synchronize(backend)
        end
        return out_res
    end
end

function ChainRulesCore.rrule(::typeof(interpolate_pure), points_to_interpolate, input_array, input_array_spacing, keep_begining_same, extrapolate_value, is_nearest_neighbour)
    output = interpolate_pure(points_to_interpolate, input_array, input_array_spacing, keep_begining_same, extrapolate_value, is_nearest_neighbour)

    function interpolate_pullback(d_output_unthunked)
        d_output_raw = unthunk(d_output_unthunked)
        d_points = zero(points_to_interpolate)
        d_input = zero(input_array)

        backend = get_backend(points_to_interpolate)
        source_arr_shape = size(input_array)

        is_batched = ndims(input_array) == 4

        # Prepare 4D helpers if batched
        points_batch_stride = 0
        spacing_arg = input_array_spacing

        if is_batched
            batch_size = size(input_array, 4)
             if ndims(points_to_interpolate) == 3 && size(points_to_interpolate, 3) == batch_size
                 points_batch_stride = 1
            elseif ndims(points_to_interpolate) == 2
                 points_batch_stride = 0
            end

            # Prepare spacing matrix
             if isa(input_array_spacing, Vector)
                # Convert vector of tuples to matrix for kernel
                # Assuming input_array_spacing is Vector{Tuple{Float64, Float64, Float64}}
                # We need Float32 matrix
                spacing_mat = Matrix{Float32}(undef, 3, batch_size)
                for b in 1:batch_size
                    spacing_mat[1, b] = Float32(input_array_spacing[b][1])
                    spacing_mat[2, b] = Float32(input_array_spacing[b][2])
                    spacing_mat[3, b] = Float32(input_array_spacing[b][3])
                end
                if backend isa KernelAbstractions.GPU
                     spacing_mat = CuArray(spacing_mat)
                end
                spacing_arg = spacing_mat
            end
        end

        # Use Enzyme directly via a wrapper (User's Pattern)
        d_input = zero(input_array)
        d_points = zero(points_to_interpolate)
        d_output = backend isa KernelAbstractions.GPU ? CuArray(unthunk(d_output_unthunked)) : Array(unthunk(d_output_unthunked))

        # Prepare safe Const arguments for GPU Enzyme
        extrapolate_value_arr = backend isa KernelAbstractions.GPU ? CuArray([Float32(extrapolate_value)]) : [Float32(extrapolate_value)]
        
        if is_batched
            function wrapper_4d(kernel, out, src_sh, src, pts, sp, kbs, ev, inn, pbs, nd)
                kernel(out, src_sh, src, pts, sp, kbs, ev, inn, pbs, ndrange=nd)
                return nothing
            end
            kernel_spec = interpolate_kernel_4d(backend)
            Enzyme.autodiff(
                Reverse,
                wrapper_4d,
                Const(kernel_spec),
                Duplicated(output, d_output),
                Const(source_arr_shape),
                Duplicated(input_array, d_input),
                Duplicated(points_to_interpolate, d_points),
                Const(spacing_arg),
                Const(keep_begining_same),
                Const(extrapolate_value_arr),
                Const(is_nearest_neighbour),
                Const(points_batch_stride),
                Const(length(output))
            )

        else
            # 3D
            spacing_arr = backend isa KernelAbstractions.GPU ? CuArray(Float32[input_array_spacing...]) : Float32[input_array_spacing...]
            function wrapper_3d(kernel, out, src_sh, src, pts, sp, kbs, ev, inn, nd)
                 kernel(out, src_sh, src, pts, sp, kbs, ev, inn, ndrange=nd)
                 return nothing
            end
            kernel_spec = interpolate_kernel(backend)
            Enzyme.autodiff(
                Reverse,
                wrapper_3d,
                Const(kernel_spec),
                Duplicated(output, d_output),
                Const(source_arr_shape),
                Duplicated(input_array, d_input),
                Duplicated(points_to_interpolate, d_points),
                Const(spacing_arr),
                Const(keep_begining_same),
                Const(extrapolate_value_arr),
                Const(is_nearest_neighbour),
                Const(length(output))
            )
        end

        return NoTangent(), d_points, d_input, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    return output, interpolate_pullback
end


"""
perform the interpolation of the set of points in a given space
"""
function interpolate_fused_affine(input_array, affine_matrices, output_size, interpolator_enum, keep_begining_same, extrapolate_value=0, center_of_rotation=nothing)
    backend = KernelAbstractions.get_backend(input_array)
    batch_size = size(input_array, 4)
    mat_batch_size = size(affine_matrices, 3)
    
    n_spatial = prod(output_size)
    total_threads = n_spatial * batch_size
    
    out_res = KernelAbstractions.zeros(backend, eltype(input_array), total_threads)
    
    source_shape = (Int32(size(input_array, 1)), Int32(size(input_array, 2)), Int32(size(input_array, 3)))
    out_size_ka = (Int32(output_size[1]), Int32(output_size[2]), Int32(output_size[3]))
    
    if center_of_rotation === nothing
        center_shift = Float32.([(s + 0.0)/2.0 for s in output_size])
    else
        center_shift = Float32.(center_of_rotation)
    end

    center_shift_tuple = (center_shift[1], center_shift[2], center_shift[3])
    
    is_nearest = (interpolator_enum == Nearest_neighbour_en)
    
    # Prepare safe Const arguments for kernels
    extrapolate_value_arr = backend isa KernelAbstractions.GPU ? CuArray([Float32(extrapolate_value)]) : [Float32(extrapolate_value)]
    center_shift_arr = backend isa KernelAbstractions.GPU ? CuArray(Float32[center_shift_tuple...]) : Float32[center_shift_tuple...]
    
    kernel = fused_affine_interpolate_kernel(backend)
    kernel(out_res, source_shape, input_array, affine_matrices, out_size_ka, center_shift_arr, keep_begining_same, extrapolate_value_arr, is_nearest, ndrange=total_threads)
    KernelAbstractions.synchronize(backend)
    
    return out_res
end

function ChainRulesCore.rrule(::typeof(interpolate_fused_affine), input_array, affine_matrices, output_size, interpolator_enum, keep_begining_same, extrapolate_value=0)
    output = interpolate_fused_affine(input_array, affine_matrices, output_size, interpolator_enum, keep_begining_same, extrapolate_value)

    function interpolate_fused_affine_pullback(d_output_unthunked)
        d_output_raw = unthunk(d_output_unthunked)
        backend = KernelAbstractions.get_backend(input_array)
        
        source_shape = (Int32(size(input_array, 1)), Int32(size(input_array, 2)), Int32(size(input_array, 3)))
        out_size_ka = (Int32(output_size[1]), Int32(output_size[2]), Int32(output_size[3]))
        center_shift = Float32.([(s + 0.0)/2.0 for s in output_size])
        center_shift_tuple = (center_shift[1], center_shift[2], center_shift[3])
        is_nearest = (interpolator_enum == Nearest_neighbour_en)
        total_threads = length(output)

        # Use Enzyme directly via a wrapper (User's Pattern)
        d_input = zero(input_array)
        d_affine = zero(affine_matrices)
        d_output = backend isa KernelAbstractions.GPU ? CuArray(d_output_raw) : Array(d_output_raw)

        function wrapper(kernel, out, src_sh, src, mat, out_sh, shift, kbs, ev, inn, nd)
            kernel(out, src_sh, src, mat, out_sh, shift, kbs, ev, inn, ndrange=nd)
            return nothing
        end

        kernel_spec = fused_affine_interpolate_kernel(backend)

        # Prepare safe Const arguments for GPU Enzyme
        extrapolate_value_arr = backend isa KernelAbstractions.GPU ? CuArray([Float32(extrapolate_value)]) : [Float32(extrapolate_value)]
        center_shift_arr = backend isa KernelAbstractions.GPU ? CuArray(Float32[center_shift_tuple...]) : Float32[center_shift_tuple...]

        Enzyme.autodiff(
            Reverse,
            wrapper,
            Const(kernel_spec),
            Duplicated(output, d_output),
            Const(source_shape),
            Duplicated(input_array, d_input),
            Duplicated(affine_matrices, d_affine),
            Const(out_size_ka),
            Const(center_shift_arr),
            Const(keep_begining_same),
            Const(extrapolate_value_arr),
            Const(is_nearest),
            Const(total_threads)
        )
        
        return NoTangent(), d_input, d_affine, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    return output, interpolate_fused_affine_pullback
end

"""
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
@kernel function fused_affine_kernel!(out, src, M, tx, ty, tz, sx, sy, sz, is_nearest)
    i = @index(Global, Linear)
    iz = (i - 1) ÷ (tx * ty) + 1
    rem_z = (i - 1) % (tx * ty)
    iy = rem_z ÷ tx + 1
    ix = rem_z % tx + 1
    m11 = M[1, 1]; m12 = M[1, 2]; m13 = M[1, 3]; m14 = M[1, 4]
    m21 = M[2, 1]; m22 = M[2, 2]; m23 = M[2, 3]; m24 = M[2, 4]
    m31 = M[3, 1]; m32 = M[3, 2]; m33 = M[3, 3]; m34 = M[3, 4]
    px = Float32(ix) * m11 + Float32(iy) * m12 + Float32(iz) * m13 + m14
    py = Float32(ix) * m21 + Float32(iy) * m22 + Float32(iz) * m23 + m24
    pz = Float32(ix) * m31 + Float32(iy) * m32 + Float32(iz) * m33 + m34
    if px < 1.0f0 || py < 1.0f0 || pz < 1.0f0 || px > Float32(sx) || py > Float32(sy) || pz > Float32(sz)
        @inbounds out[i] = 0.0f0
    else
        if is_nearest
            nx, ny, nz = round(Int, px), round(Int, py), round(Int, pz)
            nx = clamp(nx, 1, Int(sx)); ny = clamp(ny, 1, Int(sy)); nz = clamp(nx, 1, Int(sz))
            @inbounds out[i] = Float32(src[nx, ny, nz])
        else
            x0, y0, z0 = floor(Int, px), floor(Int, py), floor(Int, pz)
            dx = px - Float32(x0); dy = py - Float32(y0); dz = pz - Float32(z0)
            x1 = clamp(x0 + 1, 1, Int(sx)); y1 = clamp(y0 + 1, 1, Int(sy)); z1 = clamp(z0 + 1, 1, Int(sz))
            x0 = clamp(x0, 1, Int(sx)); y0 = clamp(y0, 1, Int(sy)); z0 = clamp(z0, 1, Int(sz))
            c000 = src[x0, y0, z0]; c100 = src[x1, y0, z0]; c010 = src[x0, y1, z0]; c110 = src[x1, y1, z0]
            c001 = src[x0, y0, z1]; c101 = src[x1, y0, z1]; c011 = src[x0, y1, z1]; c111 = src[x1, y1, z1]
            c00 = c000 * (1.0f0 - dx) + c100 * dx; c10 = c010 * (1.0f0 - dx) + c110 * dx
            c01 = c001 * (1.0f0 - dx) + c101 * dx; c11 = c011 * (1.0f0 - dx) + c111 * dx
            c0 = c00 * (1.0f0 - dy) + c10 * dy; c1 = c01 * (1.0f0 - dy) + c11 * dy
            @inbounds out[i] = c0 * (1.0f0 - dz) + c1 * dz
        end
    end
end
@kernel function fused_affine_kernel_item!(out, src, M_batch, tx, ty, tz, sx, sy, sz, is_nearest, b)
    i = @index(Global, Linear)
    iz = (i - 1) ÷ (tx * ty) + 1
    rem_z = (i - 1) % (tx * ty)
    iy = rem_z ÷ tx + 1
    ix = rem_z % tx + 1
    m11 = M_batch[1, 1, b]; m12 = M_batch[1, 2, b]; m13 = M_batch[1, 3, b]; m14 = M_batch[1, 4, b]
    m21 = M_batch[2, 1, b]; m22 = M_batch[2, 2, b]; m23 = M_batch[2, 3, b]; m24 = M_batch[2, 4, b]
    m31 = M_batch[3, 1, b]; m32 = M_batch[3, 2, b]; m33 = M_batch[3, 3, b]; m34 = M_batch[3, 4, b]
    px = Float32(ix) * m11 + Float32(iy) * m12 + Float32(iz) * m13 + m14
    py = Float32(ix) * m21 + Float32(iy) * m22 + Float32(iz) * m23 + m24
    pz = Float32(ix) * m31 + Float32(iy) * m32 + Float32(iz) * m33 + m34
    if px < 1.0f0 || py < 1.0f0 || pz < 1.0f0 || px > Float32(sx) || py > Float32(sy) || pz > Float32(sz)
        @inbounds out[ix, iy, iz, b] = 0.0f0
    else
        if is_nearest
            nx, ny, nz = round(Int, px), round(Int, py), round(Int, pz)
            nx = clamp(nx, 1, Int(sx)); ny = clamp(ny, 1, Int(sy)); nz = clamp(nz, 1, Int(sz))
            @inbounds out[ix, iy, iz, b] = Float32(src[nx, ny, nz, b])
        else
            x0, y0, z0 = floor(Int, px), floor(Int, py), floor(Int, pz)
            dx = px - Float32(x0); dy = py - Float32(y0); dz = pz - Float32(z0)
            x1 = clamp(x0 + 1, 1, Int(sx)); y1 = clamp(y0 + 1, 1, Int(sy)); z1 = clamp(z0 + 1, 1, Int(sz))
            x0 = clamp(x0, 1, Int(sx)); y0 = clamp(y0, 1, Int(sy)); z0 = clamp(z0, 1, Int(sz))
            c000 = src[x0, y0, z0, b]; c100 = src[x1, y0, z0, b]; c010 = src[x0, y1, z0, b]; c110 = src[x1, y1, z0, b]
            c001 = src[x0, y0, z1, b]; c101 = src[x1, y0, z1, b]; c011 = src[x0, y1, z1, b]; c111 = src[x1, y1, z1, b]
            c00 = c000 * (1.0f0 - dx) + c100 * dx; c10 = c010 * (1.0f0 - dx) + c110 * dx
            c01 = c001 * (1.0f0 - dx) + c101 * dx; c11 = c011 * (1.0f0 - dx) + c111 * dx
            c0 = c00 * (1.0f0 - dy) + c10 * dy; c1 = c01 * (1.0f0 - dy) + c11 * dy
            @inbounds out[ix, iy, iz, b] = c0 * (1.0f0 - dz) + c1 * dz
        end
    end
end

function fused_affine_kernel_item_launcher!(out, src, M_batch, tx, ty, tz, sx, sy, sz, is_nearest, nrange, b)
    backend = KernelAbstractions.get_backend(out)
    kernel = fused_affine_kernel_item!(backend, 256)
    kernel(out, src, M_batch, tx, ty, tz, sx, sy, sz, is_nearest, b, ndrange=nrange)
    return nothing
end
@kernel function batched_fused_affine_kernel!(out, src, M_batch, tx, ty, tz, sx, sy, sz, is_nearest)
    i_v, ib = @index(Global, Cartesian).I
    iz = (i_v - 1) ÷ (tx * ty) + 1
    rem_z = (i_v - 1) % (tx * ty)
    iy = rem_z ÷ tx + 1
    ix = rem_z % tx + 1
    m11 = M_batch[1, 1, ib]; m12 = M_batch[1, 2, ib]; m13 = M_batch[1, 3, ib]; m14 = M_batch[1, 4, ib]
    m21 = M_batch[2, 1, ib]; m22 = M_batch[2, 2, ib]; m23 = M_batch[2, 3, ib]; m24 = M_batch[2, 4, ib]
    m31 = M_batch[3, 1, ib]; m32 = M_batch[3, 2, ib]; m33 = M_batch[3, 3, ib]; m34 = M_batch[3, 4, ib]
    px = Float32(ix) * m11 + Float32(iy) * m12 + Float32(iz) * m13 + m14
    py = Float32(ix) * m21 + Float32(iy) * m22 + Float32(iz) * m23 + m24
    pz = Float32(ix) * m31 + Float32(iy) * m32 + Float32(iz) * m33 + m34
    if px < 1.0f0 || py < 1.0f0 || pz < 1.0f0 || px > Float32(sx) || py > Float32(sy) || pz > Float32(sz)
        @inbounds out[ix, iy, iz, ib] = 0.0f0
    else
        if is_nearest
            nx = Int(round(px)); ny = Int(round(py)); nz = Int(round(pz))
            nx = clamp(nx, 1, Int(sx)); ny = clamp(ny, 1, Int(sy)); nz = clamp(nz, 1, Int(sz))
            @inbounds out[ix, iy, iz, ib] = Float32(src[nx, ny, nz, ib])
        else
            x0 = floor(Int, px); y0 = floor(Int, py); z0 = floor(Int, pz)
            x1 = clamp(x0 + 1, 1, Int(sx)); y1 = clamp(y0 + 1, 1, Int(sy)); z1 = clamp(z0 + 1, 1, Int(sz))
            x0 = clamp(x0, 1, Int(sx)); y0 = clamp(y0, 1, Int(sy)); z0 = clamp(z0, 1, Int(sz))
            xd = px - Float32(x0); yd = py - Float32(y0); zd = pz - Float32(z0)
            @inbounds begin
                v000 = Float32(src[x0, y0, z0, ib]); v100 = Float32(src[x1, y0, z0, ib])
                v010 = Float32(src[x0, y1, z0, ib]); v110 = Float32(src[x1, y1, z0, ib])
                v001 = Float32(src[x0, y0, z1, ib]); v101 = Float32(src[x1, y0, z1, ib])
                v011 = Float32(src[x0, y1, z1, ib]); v111 = Float32(src[x1, y1, z1, ib])
                c00 = v000 * (1.0f0 - xd) + v100 * xd
                c10 = v010 * (1.0f0 - xd) + v110 * xd
                c01 = v001 * (1.0f0 - xd) + v101 * xd
                c11 = v011 * (1.0f0 - xd) + v111 * xd
                c0 = c00 * (1.0f0 - yd) + c10 * yd
                c1 = c01 * (1.0f0 - yd) + c11 * yd
                out[ix, iy, iz, ib] = c0 * (1.0f0 - zd) + c1 * zd
            end
        end
    end
end

function interpolate_fused_affine(src, M, target_size, interpolator_enum)
    backend = get_backend(src)
    is_batched = ndims(M) == 3
    is_nearest = (interpolator_enum == Nearest_neighbour_en)
    tx, ty, tz = Int32.(target_size)
    sx, sy, sz = Int32.(size(src)[1:3])
    if is_batched
        batch_size = size(M, 3)
        out = KernelAbstractions.zeros(backend, Float32, target_size..., batch_size)
        batched_fused_affine_kernel!(backend, 256)(out, src, M, tx, ty, tz, sx, sy, sz, is_nearest, ndrange=(prod(target_size), batch_size))
    else
        out = KernelAbstractions.zeros(backend, Float32, target_size...)
        fused_affine_kernel!(backend, 256)(out, src, M, tx, ty, tz, sx, sy, sz, is_nearest, ndrange=prod(target_size))
    end
    synchronize(backend)
    return out
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
    return nothing
end

function fused_affine_enzyme_launcher!(out, src, M, tx, ty, tz, sx, sy, sz, is_nearest, nrange)
    backend = KernelAbstractions.get_backend(out)
    kernel = fused_affine_kernel!(backend, 256)
    kernel(out, src, M, tx, ty, tz, sx, sy, sz, is_nearest, ndrange=nrange)
    return nothing
end

function batched_fused_affine_enzyme_launcher!(out, src, M_batch, tx, ty, tz, sx, sy, sz, is_nearest, n_voxels, b_size)
    backend = KernelAbstractions.get_backend(out)
    kernel = batched_fused_affine_kernel!(backend, 256)
    kernel(out, src, M_batch, tx, ty, tz, sx, sy, sz, is_nearest, ndrange=(Int(n_voxels), Int(b_size)))
    return nothing
end

# CPU specific wrapper for single subject fused affine (to avoid KA overhead in AD if possible)
function fused_affine_cpu_wrapper!(out, src, M, tx, ty, tz, sx, sy, sz, is_nearest, nrange)
    kernel = fused_affine_kernel!(KernelAbstractions.CPU(), 256)
    kernel(out, src, M, tx, ty, tz, sx, sy, sz, is_nearest, ndrange=nrange)
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

@kernel function affine_coords_kernel(points_out, @Const(affine_matrices), @Const(spatial_size), batch_size, @Const(center_shift))
    i = @index(Global, Linear)

    # Calculate output index (spatial) and batch index
    n_spatial = prod(spatial_size)

    # Map linear index to spatial index and batch index
    # We iterate over (N_points * BatchSize)
    idx_spatial = (i - 1) % n_spatial + 1
    idx_batch = (i - 1) ÷ n_spatial + 1

    # Convert spatial index to 3D coords (x,y,z)
    sx = spatial_size[1]
    sy = spatial_size[2]
    # sz = spatial_size[3]
    stride_z = sx * sy

    iz = (idx_spatial - 1) ÷ stride_z + 1
    rem_z = (idx_spatial - 1) % stride_z
    iy = rem_z ÷ sx + 1
    ix = rem_z % sx + 1

    # Center shift
    px = Float32(ix) - center_shift[1]
    py = Float32(iy) - center_shift[2]
    pz = Float32(iz) - center_shift[3]

    # Get affine matrix for this batch
    # affine_matrices is (4, 4, Batch) or (4, 4, 1)
    # If size(affine_matrices, 3) == 1, use 1, else use idx_batch
    mat_idx = size(affine_matrices, 3) == 1 ? 1 : idx_batch

    # Read matrix (column major)
    # M_inv is passed directly
    m11 = affine_matrices[1, 1, mat_idx]
    m21 = affine_matrices[2, 1, mat_idx]
    m31 = affine_matrices[3, 1, mat_idx]

    m12 = affine_matrices[1, 2, mat_idx]
    m22 = affine_matrices[2, 2, mat_idx]
    m32 = affine_matrices[3, 2, mat_idx]

    m13 = affine_matrices[1, 3, mat_idx]
    m23 = affine_matrices[2, 3, mat_idx]
    m33 = affine_matrices[3, 3, mat_idx]

    m14 = affine_matrices[1, 4, mat_idx]
    m24 = affine_matrices[2, 4, mat_idx]
    m34 = affine_matrices[3, 4, mat_idx]

    # Apply affine transform: p_new = M * p
    # p is [px, py, pz, 1]

    new_px = m11*px + m12*py + m13*pz + m14
    new_py = m21*px + m22*py + m23*pz + m24
    new_pz = m31*px + m32*py + m33*pz + m34

    # Shift back
    final_x = new_px + center_shift[1]
    final_y = new_py + center_shift[2]
    final_z = new_pz + center_shift[3]

    # Write to output (3, N_spatial, Batch)
    # Output is linearly indexed as well, or we can use 3D index
    # points_out is reshaped to (3, N_spatial * Batch) or similar?
    # Actually, let's treat points_out as linear array of size (3, N_total)
    # Layout: [x1, y1, z1, x2, y2, z2...]
    # BUT interpolate_kernel expects (3, N_points) or (3, N_points, Batch)
    # If 3D array (3, N, B):
    # points_out[1, idx_spatial, idx_batch] = final_x

    # KA handles multi-dim arrays:
    points_out[1, idx_spatial, idx_batch] = final_x
    points_out[2, idx_spatial, idx_batch] = final_y
    points_out[3, idx_spatial, idx_batch] = final_z
end

function generate_affine_coords_cpu_loop!(points_out, affine_matrices, spatial_size, batch_size, center_shift)
    n_spatial = prod(spatial_size)
    sx = spatial_size[1]
    sy = spatial_size[2]
    stride_z = sx * sy

    for i in 1:(n_spatial * batch_size)
        idx_spatial = (i - 1) % n_spatial + 1
        idx_batch = (i - 1) ÷ n_spatial + 1

        iz = (idx_spatial - 1) ÷ stride_z + 1
        rem_z = (idx_spatial - 1) % stride_z
        iy = rem_z ÷ sx + 1
        ix = rem_z % sx + 1

        px = Float32(ix) - center_shift[1]
        py = Float32(iy) - center_shift[2]
        pz = Float32(iz) - center_shift[3]

        mat_idx = size(affine_matrices, 3) == 1 ? 1 : idx_batch

        new_px = affine_matrices[1, 1, mat_idx]*px + affine_matrices[1, 2, mat_idx]*py + affine_matrices[1, 3, mat_idx]*pz + affine_matrices[1, 4, mat_idx]
        new_py = affine_matrices[2, 1, mat_idx]*px + affine_matrices[2, 2, mat_idx]*py + affine_matrices[2, 3, mat_idx]*pz + affine_matrices[2, 4, mat_idx]
        new_pz = affine_matrices[3, 1, mat_idx]*px + affine_matrices[3, 2, mat_idx]*py + affine_matrices[3, 3, mat_idx]*pz + affine_matrices[3, 4, mat_idx]

        points_out[1, idx_spatial, idx_batch] = new_px + center_shift[1]
        points_out[2, idx_spatial, idx_batch] = new_py + center_shift[2]
        points_out[3, idx_spatial, idx_batch] = new_pz + center_shift[3]
    end
end

function generate_affine_coords(spatial_size, affine_matrices, backend)
    # affine_matrices: Array{Float32, 3} of size (4, 4, Batch) or (4, 4, 1)
    # Returns points_to_interpolate: Array{Float32, 3} of size (3, N_points, Batch)

    batch_size = size(affine_matrices, 3)
    # If matrices are shared (size 1) but we want to generate points for a batch?
    # Usually if shared matrix, we can generate points once (3, N, 1).
    # But interpolate_kernel_4d handles batch stride.
    # If we want unique output per batch (e.g. if we had unique shift or something), we need (3, N, B).
    # Here, we assume if affine_matrices has B > 1, output has B.
    # If affine_matrices has B == 1, output CAN be B=1.

    # BUT wait, the caller might want B outputs even if matrix is shared (e.g. if other params differ).
    # However, for pure affine transform, if matrix is shared, points are shared.

    out_batch_size = batch_size
    n_points = prod(spatial_size)

    points_out = KernelAbstractions.zeros(backend, Float32, 3, n_points, out_batch_size)

    center_shift = Float32.([(s + 0.0)/2.0 for s in spatial_size])
    # Move center_shift to GPU? It's small, can be captured as Const tuple or array?
    # KA usually handles small arrays in arguments fine, or use Tuple.
    center_shift_tuple = (center_shift[1], center_shift[2], center_shift[3])

    # Launch kernel
    # Total threads: n_points * out_batch_size
    affine_coords_kernel(backend, 256)(points_out, affine_matrices, spatial_size, out_batch_size, center_shift_tuple, ndrange=n_points * out_batch_size)
    synchronize(backend)

    return points_out
end

function ChainRulesCore.rrule(::typeof(generate_affine_coords), spatial_size, affine_matrices, backend)
    output = generate_affine_coords(spatial_size, affine_matrices, backend)

    function generate_affine_coords_pullback(d_output_unthunked)
        d_output = unthunk(d_output_unthunked)
        d_affine_matrices = zero(affine_matrices)

        center_shift = Float32.([(s + 0.0)/2.0 for s in spatial_size])
        center_shift_tuple = (center_shift[1], center_shift[2], center_shift[3])
        batch_size = size(affine_matrices, 3)

        if backend isa KernelAbstractions.CPU
            function cpu_wrapper(out, mats, sz, bs, cs)
                generate_affine_coords_cpu_loop!(out, mats, sz, bs, cs)
                return nothing
            end

            Enzyme.autodiff(
                Reverse,
                cpu_wrapper,
                Const,
                Duplicated(output, d_output),
                Duplicated(affine_matrices, d_affine_matrices),
                Const(spatial_size),
                Const(batch_size),
                Const(center_shift_tuple)
            )
            return NoTangent(), NoTangent(), d_affine_matrices, NoTangent()
        else
            # GPU path - Use CPU fallback for stability
            out_cpu = Array(output)
            d_out_cpu = Array(unthunk(d_output_unthunked))
            mats_cpu = Array(affine_matrices)
            
            d_mats_cpu = zero(mats_cpu)
            
            function fallback_gpu(out, mats, sz, bs, cs)
                generate_affine_coords_cpu_loop!(out, mats, sz, bs, cs)
                return nothing
            end

            Enzyme.autodiff(
                Reverse,
                fallback_gpu,
                Const,
                Duplicated(out_cpu, d_out_cpu),
                Duplicated(mats_cpu, d_mats_cpu),
                Const(spatial_size),
                Const(batch_size),
                Const(center_shift_tuple)
            )
            
            return NoTangent(), NoTangent(), CuArray(d_mats_cpu), NoTangent()
        end
    end

    return output, generate_affine_coords_pullback
end

function create_batched_medimage(med_images::Vector{MedImage})::BatchedMedImage
    if isempty(med_images)
        error("Input vector of MedImages is empty")
    end

    # Check for consistency in image size (spatial dimensions must match for stacking)
    first_size = size(med_images[1].voxel_data)
    for i in 2:length(med_images)
        if size(med_images[i].voxel_data) != first_size
            error("All images in the batch must have the same spatial dimensions. Image 1 has $first_size, Image $i has $(size(med_images[i].voxel_data))")
        end
    end

    # Stack voxel data
    # Assuming voxel_data is 3D (x, y, z), stacking along 4th dim
    voxel_data_batch = cat([img.voxel_data for img in med_images]...; dims=4)

    return BatchedMedImage(
        voxel_data = voxel_data_batch,
        origin = [img.origin for img in med_images],
        spacing = [img.spacing for img in med_images],
        direction = [img.direction for img in med_images],
        image_type = [img.image_type for img in med_images],
        image_subtype = [img.image_subtype for img in med_images],
        date_of_saving = [img.date_of_saving for img in med_images],
        acquistion_time = [img.acquistion_time for img in med_images],
        patient_id = [img.patient_id for img in med_images],
        current_device = med_images[1].current_device, # Assume all on same device
        study_uid = [img.study_uid for img in med_images],
        patient_uid = [img.patient_uid for img in med_images],
        series_uid = [img.series_uid for img in med_images],
        study_description = [img.study_description for img in med_images],
        legacy_file_name = [img.legacy_file_name for img in med_images],
        display_data = [img.display_data for img in med_images],
        clinical_data = [img.clinical_data for img in med_images],
        is_contrast_administered = [img.is_contrast_administered for img in med_images],
        metadata = [img.metadata for img in med_images]
    )
end

function ChainRulesCore.rrule(::typeof(create_batched_medimage), med_images::Vector{MedImage})
    y = create_batched_medimage(med_images)
    function create_batched_pullback(d_y)
        d_y_unthunked = unthunk(d_y)
        # Check if we have gradients for voxel_data
        # d_y_unthunked should be Tangent{BatchedMedImage}
        # access .voxel_data

        d_voxels = d_y_unthunked.voxel_data

        if d_voxels isa AbstractArray
            d_med_images = map(1:length(med_images)) do i
                # Create a Tangent for MedImage with just voxel_data
                # We slice the 4th dimension
                slice = selectdim(d_voxels, 4, i)
                # Tangent takes kwargs for fields
                Tangent{MedImage}(; voxel_data=slice)
            end
            return NoTangent(), d_med_images
        else
             return NoTangent(), NoTangent()
        end
    end
    return y, create_batched_pullback
end

function unbatch_medimage(batched_image::BatchedMedImage)::Vector{MedImage}
    batch_size = size(batched_image.voxel_data, 4)
    med_images = Vector{MedImage}(undef, batch_size)

    for i in 1:batch_size
        # Extract 3D slice
        voxel_slice = selectdim(batched_image.voxel_data, 4, i)
        # Copy to ensure it's standard array and owns its memory (not just a view)
        voxel_slice = copy(voxel_slice)

        med_images[i] = MedImage(
            voxel_data = voxel_slice,
            origin = batched_image.origin[i],
            spacing = batched_image.spacing[i],
            direction = batched_image.direction[i],
            image_type = batched_image.image_type[i],
            image_subtype = batched_image.image_subtype[i],
            date_of_saving = batched_image.date_of_saving[i],
            acquistion_time = batched_image.acquistion_time[i],
            patient_id = batched_image.patient_id[i],
            current_device = batched_image.current_device,
            study_uid = batched_image.study_uid[i],
            patient_uid = batched_image.patient_uid[i],
            series_uid = batched_image.series_uid[i],
            study_description = batched_image.study_description[i],
            legacy_file_name = batched_image.legacy_file_name[i],
            display_data = batched_image.display_data[i],
            clinical_data = batched_image.clinical_data[i],
            is_contrast_administered = batched_image.is_contrast_administered[i],
            metadata = batched_image.metadata[i]
        )
    end
    return med_images
end

function ChainRulesCore.rrule(::typeof(interpolate_fused_affine), src, M, target_size, interpolator_enum)
    output = interpolate_fused_affine(src, M, target_size, interpolator_enum)

    function interpolate_pullback(d_output_unthunked)
        d_output_raw = unthunk(d_output_unthunked)
        d_src = zero(src)
        d_M = zero(M)

        backend = get_backend(output)
        is_batched = ndims(M) == 3
        is_nearest = (interpolator_enum == Nearest_neighbour_en)

        tx, ty, tz = Int32.(target_size)
        sx, sy, sz = Int32.(size(src)[1:3])
        ndrange_single = prod(target_size)
        
        if backend isa KernelAbstractions.CPU
            tx, ty, tz = Int32.(target_size)
            sx, sy, sz = Int32.(size(src)[1:3])
            ndrange_single = prod(target_size)

            # Enzyme CPU often fails on batched views or complex closures.
            # For CPU batched, we provide a simpler path if possible, or just the single-subject one if not batched.
            if !is_batched
                Enzyme.autodiff(Reverse, fused_affine_cpu_wrapper!, Const,
                    Duplicated(output, d_output_raw),
                    Duplicated(src, d_src),
                    Duplicated(M, d_M),
                    Const(tx), Const(ty), Const(tz), Const(sx), Const(sy), Const(sz), 
                    Const(is_nearest), Const(Int(ndrange_single)))
            else
                # For batched CPU, we currently don't have a stable high-perf Enzyme path 
                # that doesn't produce 'iterate' errors. 
                # We could implement a manual loop but that might be slow.
                # However, for verification we can use a loop.
                for b in 1:size(M, 3)
                    # We use itemized kernel with NO views passed to Enzyme
                    Enzyme.autodiff(Reverse, fused_affine_kernel_item_launcher!, Const,
                        Duplicated(output, d_output_raw),
                        Duplicated(src, d_src),
                        Duplicated(M, d_M),
                        Const(tx), Const(ty), Const(tz), Const(sx), Const(sy), Const(sz), 
                        Const(is_nearest), Const(Int(ndrange_single)), Const(Int(b)))
                end
            end
        else
            # GPU backward pass via looping pullback over batch
            # Ensure gradients are on GPU
            d_out = is_cuda_array(output) && !is_cuda_array(d_output_raw) ? CuArray(d_output_raw) : d_output_raw
            d_src = is_cuda_array(src) && !is_cuda_array(d_src) ? CuArray(d_src) : d_src
            d_M = is_cuda_array(M) && !is_cuda_array(d_M) ? CuArray(d_M) : d_M

            if is_batched
                batch_size = size(M, 3)
                for b in 1:batch_size
                    Enzyme.autodiff(Reverse, fused_affine_kernel_item_launcher!,
                        Duplicated(output, d_out),
                        Duplicated(src, d_src),
                        Duplicated(M, d_M),
                        Const(tx), Const(ty), Const(tz), Const(sx), Const(sy), Const(sz), 
                        Const(is_nearest),
                        Const(Int(ndrange_single)),
                        Const(Int(b)))
                end
            else
                Enzyme.autodiff(Reverse, fused_affine_enzyme_launcher!,
                    Duplicated(output, d_out),
                    Duplicated(src, d_src),
                    Duplicated(M, d_M),
                    Const(tx), Const(ty), Const(tz), Const(sx), Const(sy), Const(sz), 
                    Const(is_nearest),
                    Const(Int(ndrange_single)))
            end
            synchronize(backend)
        end

        # Return gradients
        d_src_out = is_cuda_array(d_src) ? Array(d_src) : d_src
        d_M_out = is_cuda_array(d_M) ? Array(d_M) : d_M
        
        return NoTangent(), d_src_out, d_M_out, NoTangent(), NoTangent()
    end
    return output, interpolate_pullback
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

            function wrapper(launcher, out, img, osp, nsp, nd, sd, nrd)
                launcher(out, img, osp, nsp, nd, sd, nrd)
                return nothing
            end

            if interpolator_enum == Nearest_neighbour_en
                Enzyme.autodiff(
                    Reverse,
                    wrapper,
                    Const(nearest_enzyme_launcher!),
                    Duplicated(vec(output), vec(d_output)),
                    Duplicated(image_data, d_image),
                    Duplicated(old_spacing_arr, d_osp),
                    Duplicated(new_spacing_arr, d_nsp),
                    Duplicated(new_dims_arr, d_ndims),
                    Duplicated(src_dims_arr, d_src_dims),
                    Const(ndrange_val)
                )
            else
                function trilinear_enzyme_launcher!(out, img, osp_arr, nsp_arr, ndims_arr, src_dims_arr, ndrange_val)
                    backend = KernelAbstractions.get_backend(out)
                    kernel = trilinear_resample_enzyme_kernel!(backend, 256)
                    kernel(out, img, osp_arr, nsp_arr, ndims_arr, src_dims_arr, ndrange=ndrange_val)
                    return nothing
                end
                Enzyme.autodiff(
                    Reverse,
                    wrapper,
                    Const(trilinear_enzyme_launcher!),
                    Duplicated(vec(output), vec(d_output)),
                    Duplicated(image_data, d_image),
                    Duplicated(old_spacing_arr, d_osp),
                    Duplicated(new_spacing_arr, d_nsp),
                    Duplicated(new_dims_arr, d_ndims),
                    Duplicated(src_dims_arr, d_src_dims),
                    Const(ndrange_val)
                )
            end
            synchronize(backend)
        end

        # Convert gradients back to CPU if needed for Zygote
        d_image_out = is_cuda_array(d_image) ? Array(d_image) : d_image
        return NoTangent(), d_image_out, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    return output, resample_pullback
end

end#Utils
