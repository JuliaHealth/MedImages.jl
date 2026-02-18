
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
    A_x1 = 1:input_array_spacing[1]:(1+input_array_spacing[1]*(old_size[1]-1))
    A_x2 = 1:input_array_spacing[2]:(1+input_array_spacing[2]*(old_size[2]-1))
    A_x3 = 1:input_array_spacing[3]:(1+input_array_spacing[3]*(old_size[3]-1))


    itp = extrapolate(itp, extrapolate_value)
    itp = scale(itp, A_x1, A_x2, A_x3)
    
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

            c0 * (1.0f0 - zd) + c1 * zd
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

    out_batch_size = batch_size
    n_points = prod(spatial_size)

    points_out = KernelAbstractions.zeros(backend, Float32, 3, n_points, out_batch_size)

    center_shift = Float32.([(s + 0.0)/2.0 for s in spatial_size])
    center_shift_tuple = (center_shift[1], center_shift[2], center_shift[3])

    # Launch kernel
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
