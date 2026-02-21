
@kernel function fused_affine_enzyme_kernel!(out_res, source_arr_shape, source_arr, affine_matrices, output_size, center_shift)
    I = @index(Global)
    out_res[I] = 0.0f0
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
