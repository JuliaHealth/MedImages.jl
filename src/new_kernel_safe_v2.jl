
@kernel function fused_affine_enzyme_kernel!(out_res, source_arr_shape, source_arr, affine_matrices, output_size, center_shift)
    # Map index to output voxel and batch
    # ndrange is (prod(output_size) * BatchSize)
    I = @index(Global)
    
    n_spatial = output_size[1] * output_size[2] * output_size[3]
    idx_spatial = (I - 1) % n_spatial + 1
    idx_batch = (I - 1) ÷ n_spatial + 1
    
    # Map spatial index to (ix, iy, iz) 1-based
    stride_z = output_size[1] * output_size[2]
    iz = (idx_spatial - 1) ÷ stride_z + 1
    rem_z = (idx_spatial - 1) % stride_z
    iy = rem_z ÷ output_size[1] + 1
    ix = rem_z % output_size[1] + 1
    
    # Shift to center
    px = Float32(ix) - center_shift[1]
    py = Float32(iy) - center_shift[2]
    pz = Float32(iz) - center_shift[3]
    
    # Which matrix to use?
    # affine_matrices is (4, 4, Batch) or (4, 4, 1)
    mat_idx = idx_batch
    
    # Apply inverse affine matrix: p_source = M_inv * p_output
    @inbounds begin
        new_px = affine_matrices[1,1,mat_idx]*px + affine_matrices[1,2,mat_idx]*py + affine_matrices[1,3,mat_idx]*pz + affine_matrices[1,4,mat_idx]
        new_py = affine_matrices[2,1,mat_idx]*px + affine_matrices[2,2,mat_idx]*py + affine_matrices[2,3,mat_idx]*pz + affine_matrices[2,4,mat_idx]
        new_pz = affine_matrices[3,1,mat_idx]*px + affine_matrices[3,2,mat_idx]*py + affine_matrices[3,3,mat_idx]*pz + affine_matrices[3,4,mat_idx]
        
        # Shift back to find real index in source
        real_x = new_px + center_shift[1]
        real_y = new_py + center_shift[2]
        real_z = new_pz + center_shift[3]
        
        # Bounds check
        sx = Int(source_arr_shape[1])
        sy = Int(source_arr_shape[2])
        sz = Int(source_arr_shape[3])

        if isnan(real_x) || isnan(real_y) || isnan(real_z)
             out_res[I] = 0.0f0
        elseif real_x < 1.0f0 || real_y < 1.0f0 || real_z < 1.0f0 || real_x > Float32(sx) || real_y > Float32(sy) || real_z > Float32(sz)
            out_res[I] = 0.0f0
        else
            # Trilinear Interpolation - FULLY UNROLLED
            # Use floor(Int, ...) to safely convert float to int index
            x0 = floor(Int, real_x)
            y0 = floor(Int, real_y)
            z0 = floor(Int, real_z)

            # x1 = min(x0 + 1, sx) -> if x0+1 > sx ? sx : x0+1
            x1 = x0 + 1
            if x1 > sx
                x1 = sx
            end

            y1 = y0 + 1
            if y1 > sy
                y1 = sy
            end
            
            z1 = z0 + 1
            if z1 > sz
                z1 = sz
            end

            xd = real_x - x0
            yd = real_y - y0
            zd = real_z - z0
            
            # Unrolled load and calc
            v000 = Float32(source_arr[x0, y0, z0, idx_batch])
            v100 = Float32(source_arr[x1, y0, z0, idx_batch])
            
            v010 = Float32(source_arr[x0, y1, z0, idx_batch])
            v110 = Float32(source_arr[x1, y1, z0, idx_batch])
            
            v001 = Float32(source_arr[x0, y0, z1, idx_batch])
            v101 = Float32(source_arr[x1, y0, z1, idx_batch])
            
            v011 = Float32(source_arr[x0, y1, z1, idx_batch])
            v111 = Float32(source_arr[x1, y1, z1, idx_batch])

            c00 = v000 * (1.0f0 - xd) + v100 * xd
            c10 = v010 * (1.0f0 - xd) + v110 * xd
            c01 = v001 * (1.0f0 - xd) + v101 * xd
            c11 = v011 * (1.0f0 - xd) + v111 * xd

            c0 = c00 * (1.0f0 - yd) + c10 * yd
            c1 = c01 * (1.0f0 - yd) + c11 * yd

            out_res[I] = c0 * (1.0f0 - zd) + c1 * zd
        end
    end
end

function fused_affine_enzyme_launcher!(out_res, source_arr_shape_arr, source_arr, affine_matrices, output_size_arr, center_shift_arr, ndrange_val)
    backend = KernelAbstractions.get_backend(out_res)
    kernel = fused_affine_enzyme_kernel!(backend, 256)
    kernel(out_res, source_arr_shape_arr, source_arr, affine_matrices, output_size_arr, center_shift_arr, ndrange=ndrange_val)
    KernelAbstractions.synchronize(backend)
    return nothing
end

function interpolate_fused_affine(input_array, affine_matrices, output_size, interpolator_enum, keep_begining_same, extrapolate_value=0)
    batch_size = size(input_array, 4)
    out_dims = (output_size[1], output_size[2], output_size[3], batch_size)
    
    backend = get_backend(input_array)
    output = KernelAbstractions.allocate(backend, Float32, out_dims)
    
    # Prepare constants
    src_dims = size(input_array)
    source_arr_shape_arr = CuArray(Int32[src_dims[1], src_dims[2], src_dims[3]])
    output_size_arr = CuArray(Int32[output_size[1], output_size[2], output_size[3]])
    center_shift = Float32.([(s + 0.0)/2.0 for s in output_size])
    center_shift_arr = CuArray(center_shift)
    
    total_threads = prod(output_size) * batch_size
    
    if backend isa KernelAbstractions.CPU
        error("CPU path not implemented in fused kernel yet")
    else
        fused_affine_enzyme_launcher!(output, source_arr_shape_arr, input_array, affine_matrices, output_size_arr, center_shift_arr, total_threads)
    end
    
    return output
end

function ChainRulesCore.rrule(::typeof(interpolate_fused_affine), input_array, affine_matrices, output_size, interpolator_enum, keep_begining_same, extrapolate_value=0)
    output = interpolate_fused_affine(input_array, affine_matrices, output_size, interpolator_enum, keep_begining_same, extrapolate_value)
    
    function interpolate_fused_affine_pullback(d_output_unthunked)
        d_output_raw = unthunk(d_output_unthunked)
        backend = get_backend(input_array)
        if backend isa KernelAbstractions.CPU
             error("CPU autodiff for interpolate_fused_affine not implemented yet. Use GPU.")
        else
             d_input_array = zero(input_array)
             d_affine_matrices = zero(affine_matrices)
             d_output = is_cuda_array(output) && !is_cuda_array(d_output_raw) ? CuArray(d_output_raw) : d_output_raw
             
             src_dims = size(input_array)
             source_arr_shape_arr = CuArray(Int32[src_dims[1], src_dims[2], src_dims[3]])
             output_size_arr = CuArray(Int32[output_size[1], output_size[2], output_size[3]])
             center_shift = Float32.([(s + 0.0)/2.0 for s in output_size])
             center_shift_arr = CuArray(center_shift)
             
             total_threads = prod(output_size) * size(input_array, 4)
             
             Enzyme.autodiff(
                 Reverse,
                 fused_affine_enzyme_launcher!,
                 Duplicated(output, d_output),
                 Const(source_arr_shape_arr),
                 Duplicated(input_array, d_input_array),
                 Duplicated(affine_matrices, d_affine_matrices),
                 Const(output_size_arr),
                 Const(center_shift_arr),
                 Const(total_threads)
             )
             return NoTangent(), d_input_array, d_affine_matrices, NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
    end
    return output, interpolate_fused_affine_pullback
end
