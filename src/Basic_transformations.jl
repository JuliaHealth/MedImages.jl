module Basic_transformations

using CoordinateTransformations, Interpolations, StaticArrays, LinearAlgebra, Rotations, Dictionaries
using LinearAlgebra
using ImageTransformations
using ChainRulesCore
using ..MedImage_data_struct
using ..MedImage_data_struct: Nearest_neighbour_en, Linear_en, B_spline_en
using ..Load_and_save: update_voxel_data, update_voxel_and_spatial_data
using ..Utils: interpolate_my, generate_affine_coords, is_cuda_array
using KernelAbstractions

export rotate_mi, crop_mi, pad_mi, translate_mi, scale_mi, computeIndexToPhysicalPointMatrices_Julia, transformIndexToPhysicalPoint_Julia, get_voxel_center_Julia, get_real_center_Julia, Rodrigues_rotation_matrix, crop_image_around_center
export affine_transform_mi, create_affine_matrix, compose_affine_matrices

function computeIndexToPhysicalPointMatrices_Julia(im::MedImage)::Matrix{Float64}
  VImageDimension = length(im.spacing)
  spacing_vector = collect(im.spacing)
  if any(spacing_vector .== 0.0)
    error("A spacing of 0 is not allowed")
  end

  direction_matrix = reshape(collect(im.direction), VImageDimension, VImageDimension)

  if det(direction_matrix) == 0.0
    error("Bad direction, determinant is 0")
  end

  scaleMatrix = Diagonal(spacing_vector)

  indexToPhysicalPoint = direction_matrix * scaleMatrix
  return indexToPhysicalPoint
end


function transformIndexToPhysicalPoint_Julia(im::MedImage, index::Tuple{Vararg{Int}})::Tuple{Vararg{Float64}}
  indexToPhysicalPoint = computeIndexToPhysicalPointMatrices_Julia(im)
  VImageDimension = length(index)
  point = zeros(Float64, VImageDimension)

  for i in 1:VImageDimension
    point[i] = im.origin[i] + sum(indexToPhysicalPoint[i, j] * index[j] for j in 1:VImageDimension)
  end

  return Tuple(point)
end


function get_voxel_center_Julia(image::AbstractArray{T,3})::Tuple{Vararg{Float64}} where {T}
  real_size = size(image)
  real_origin = (0, 0, 0)
  return Tuple((real_size .+ real_origin) ./ 2)
end

# Geometric computations are not differentiable - they depend on image shape not voxel values
ChainRulesCore.@non_differentiable get_voxel_center_Julia(::Any)

function get_real_center_Julia(im::MedImage)::Tuple{Vararg{Float64}}
  real_size = transformIndexToPhysicalPoint_Julia(im, size(im.voxel_data))
  real_origin = transformIndexToPhysicalPoint_Julia(im, (0, 0, 0))
  return Tuple((real_size .+ real_origin) ./ 2)
end

ChainRulesCore.@non_differentiable get_real_center_Julia(::Any)
ChainRulesCore.@non_differentiable computeIndexToPhysicalPointMatrices_Julia(::Any)
ChainRulesCore.@non_differentiable transformIndexToPhysicalPoint_Julia(::Any, ::Any)

function Rodrigues_rotation_matrix(direction::NTuple{9,Float64}, axis::Int, angle::Float64)::Matrix{Float64}
  img_direction = direction
  axis_angle = if axis == 3
    (img_direction[9], img_direction[6], img_direction[3])
  elseif axis == 2
    (img_direction[8], img_direction[5], img_direction[2])
  elseif axis == 1
    (img_direction[7], img_direction[4], img_direction[1])

  end

  ux, uy, uz = axis_angle
  theta = deg2rad(angle)
  c = cos(theta)
  s = sin(theta)
  ci = 1.0 - c
  R = [ci*ux*ux+c ci*ux*uy-uz*s ci*ux*uz+uy*s;
    ci*uy*ux+uz*s ci*uy*uy+c ci*uy*uz-ux*s;
    ci*uz*ux-uy*s ci*uz*uy+ux*s ci*uz*uz+c]
  return R
end

function Rodrigues_rotation_matrix(image::MedImage, axis::Int, angle::Float64)::Matrix{Float64}
    return Rodrigues_rotation_matrix(image.direction, axis, angle)
end

# ChainRulesCore.@non_differentiable Rodrigues_rotation_matrix(::Any, ::Any, ::Any)

function crop_image_around_center(image::AbstractArray{T,3}, new_dims::Tuple{Int,Int,Int}, center::Tuple{Int,Int,Int}) where {T}
  start_z = max(1, center[1] - new_dims[1] ÷ 2)
  end_z = min(size(image, 1), start_z + new_dims[1] - 1)

  start_y = max(1, center[2] - new_dims[2] ÷ 2)
  end_y = min(size(image, 2), start_y + new_dims[2] - 1)

  start_x = max(1, center[3] - new_dims[3] ÷ 2)
  end_x = min(size(image, 3), start_x + new_dims[3] - 1)
  cropped_image = image[start_z:end_z, start_y:end_y, start_x:end_x]
  return cropped_image
end


# Helper to build rotation transformation - non-differentiable (purely geometric)
function build_rotation_transform(image::MedImage, axis::Int, angle::Float64)
  R = Rodrigues_rotation_matrix(image, axis, angle)
  v_center = collect(get_voxel_center_Julia(image.voxel_data))
  rotation_transformation = LinearMap(RotXYZ(R))
  translation = Translation(v_center...)
  transkation_center = Translation(-v_center...)
  return translation ∘ rotation_transformation ∘ transkation_center
end
ChainRulesCore.@non_differentiable build_rotation_transform(::Any, ::Any, ::Any)

function rotate_mi(image::MedImage, axis::Int, angle::Float64, Interpolator::Interpolator_enum, crop::Bool=true)::MedImage
  combined_transformation = build_rotation_transform(image, axis, angle)

  img = image.voxel_data

  if crop
      out_size = size(img)
      indices = CartesianIndices(out_size)
      indices_vec = vec(collect(indices))

      points_vec = map(idx -> begin
          pt = [Float64(idx[1]), Float64(idx[2]), Float64(idx[3])]
          combined_transformation(pt)
      end, indices_vec)

      points_to_interpolate = reduce(hcat, points_vec)

      # Interpolate
      # Use spacing (1,1,1) so that points are treated as indices
      resampled_flat = interpolate_my(points_to_interpolate, img, (1.0,1.0,1.0), Interpolator, false, 0.0, true)

      resampled_image = reshape(resampled_flat, out_size)

      image = update_voxel_data(image, resampled_image)
  else
      error("rotate_mi with crop=false is not fully supported in this patch")
  end

  return image
end


function crop_mi(im::MedImage, crop_beg::Tuple{Int64,Int64,Int64}, crop_size::Tuple{Int64,Int64,Int64}, Interpolator::Interpolator_enum)::MedImage

  julia_beg = crop_beg .+ 1
  cropped_voxel_data = @view im.voxel_data[julia_beg[1]:(julia_beg[1]+crop_size[1]-1), julia_beg[2]:(julia_beg[2]+crop_size[2]-1), julia_beg[3]:(julia_beg[3]+crop_size[3]-1)]

  dir_diag = [im.direction[1], im.direction[5], im.direction[9]]
  cropped_origin = im.origin .+ (im.spacing .* crop_beg .* dir_diag)

  cropped_im = update_voxel_and_spatial_data(im, cropped_voxel_data, cropped_origin, im.spacing, im.direction)

  return cropped_im

end

function pad_dim(arr, dim, l, r, val)
   if l==0 && r==0 return arr end
   sz = size(arr)

   left_shape = ntuple(i -> i == dim ? l : sz[i], length(sz))
   # generic fill using similar to preserve backend
   left_block = similar(arr, left_shape)
   fill!(left_block, convert(eltype(arr), val))

   right_shape = ntuple(i -> i == dim ? r : sz[i], length(sz))
   right_block = similar(arr, right_shape)
   fill!(right_block, convert(eltype(arr), val))

   cat(left_block, arr, right_block; dims=dim)
end

function ChainRulesCore.rrule(::typeof(pad_dim), arr, dim, l, r, val)
    y = pad_dim(arr, dim, l, r, val)
    function pad_dim_pullback(d_y)
        d_y_unthunked = unthunk(d_y)

        if d_y_unthunked isa ChainRulesCore.AbstractZero
             return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        # d_arr is the central part of d_y corresponding to arr
        # d_val is the sum of gradients in the padded regions

        sz = size(arr)
        # Construct ranges for central part
        inds = ntuple(i -> i == dim ? (l+1 : l+sz[i]) : (:), length(sz))
        d_arr = d_y_unthunked[inds...]

        # Calculate d_val
        # Padded regions: left (1:l) and right (end-r+1:end) along dim
        d_val = zero(val)

        if l > 0
            left_inds = ntuple(i -> i == dim ? (1:l) : (:), length(sz))
            s = sum(d_y_unthunked[left_inds...])
            d_val += s
        end
        if r > 0
            right_inds = ntuple(i -> i == dim ? (size(d_y_unthunked, dim)-r+1 : size(d_y_unthunked, dim)) : (:), length(sz))
            s = sum(d_y_unthunked[right_inds...])
            d_val += s
        end

        return NoTangent(), d_arr, NoTangent(), NoTangent(), NoTangent(), d_val
    end
    return y, pad_dim_pullback
end

function pad_mi(im::MedImage, pad_beg::Tuple{Int64,Int64,Int64}, pad_end::Tuple{Int64,Int64,Int64}, pad_val, Interpolator::Interpolator_enum)::MedImage

  data = im.voxel_data
  data = pad_dim(data, 1, pad_beg[1], pad_end[1], pad_val)
  data = pad_dim(data, 2, pad_beg[2], pad_end[2], pad_val)
  data = pad_dim(data, 3, pad_beg[3], pad_end[3], pad_val)

  dir_diag = [im.direction[1], im.direction[5], im.direction[9]]
  padded_origin = im.origin .- (im.spacing .* pad_beg .* dir_diag)

  padded_im = update_voxel_and_spatial_data(im, data, padded_origin, im.spacing, im.direction)
  return padded_im
end


function translate_mi(im::MedImage, translate_by::Int64, translate_in_axis::Int64, Interpolator::Interpolator_enum)::MedImage
  origin_val = im.origin[translate_in_axis] + translate_by * im.spacing[translate_in_axis]
  translated_origin = ntuple(i -> i == translate_in_axis ? origin_val : im.origin[i], 3)
  translated_im = update_voxel_and_spatial_data(im, im.voxel_data, translated_origin, im.spacing, im.direction)
  return translated_im
end


# Helper to build scale interpolation points - non-differentiable (purely geometric)
function build_scale_points(old_size, scale_tuple)
  new_size = Tuple(round.(Int, old_size .* scale_tuple))
  n_points = prod(new_size)
  points_to_interpolate = Matrix{Float64}(undef, 3, n_points)

  idx = 1
  for k in 1:new_size[3], j in 1:new_size[2], i in 1:new_size[1]
    # Map new coordinate to old coordinate
    points_to_interpolate[1, idx] = (i - 1.0) / scale_tuple[1] + 1.0
    points_to_interpolate[2, idx] = (j - 1.0) / scale_tuple[2] + 1.0
    points_to_interpolate[3, idx] = (k - 1.0) / scale_tuple[3] + 1.0
    idx += 1
  end

  return points_to_interpolate, new_size
end
ChainRulesCore.@non_differentiable build_scale_points(::Any, ::Any)

function scale_mi(im::MedImage, scale::Union{Float64, Tuple{Float64,Float64,Float64}}, Interpolator::Interpolator_enum)::MedImage

  scale_tuple = scale isa Float64 ? (scale, scale, scale) : scale
  old_size = size(im.voxel_data)

  # Use helper to build points (non-differentiable)
  points_to_interpolate, new_size = build_scale_points(old_size, scale_tuple)

  # Use our differentiable interpolation
  resampled_flat = interpolate_my(points_to_interpolate, im.voxel_data, (1.0, 1.0, 1.0), Interpolator, false, 0.0, true)
  new_data = reshape(resampled_flat, new_size)

  new_im = update_voxel_and_spatial_data(im, new_data, im.origin, im.spacing, im.direction)

  return new_im
end

function rotate_mi(image::BatchedMedImage, axis::Int, angle::Union{Float64, AbstractVector{Float64}}, Interpolator::Interpolator_enum, crop::Bool=true)::BatchedMedImage
    if !crop
         error("rotate_mi with crop=false is not fully supported in batched mode currently")
    end

    batch_size = size(image.voxel_data, 4)
    if angle isa AbstractVector && length(angle) != batch_size
        error("Angle vector length must match batch size")
    end

    # Construct Affine Matrices (Rotation)
    # R is rotation around 0 (because affine transform logic centers the image)
    matrices = Vector{Matrix{Float64}}(undef, batch_size)
    for b in 1:batch_size
        current_angle = (angle isa AbstractVector) ? angle[b] : angle
        current_direction = image.direction[b]
        R_sub = Rodrigues_rotation_matrix(current_direction, axis, current_angle)

        M = Matrix{Float64}(I, 4, 4)
        M[1:3, 1:3] = R_sub
        matrices[b] = M
    end

    # Use affine_transform_mi
    return affine_transform_mi(image, matrices, Interpolator)
end

function scale_mi(image::BatchedMedImage, scale::Union{Float64, Tuple{Float64,Float64,Float64}, Vector{<:Union{Float64, Tuple{Float64,Float64,Float64}}}}, Interpolator::Interpolator_enum)::BatchedMedImage
  batch_size = size(image.voxel_data, 4)
  old_size = size(image.voxel_data)[1:3]

  # Determine output size and scale matrices
  new_sizes = []
  matrices = Vector{Matrix{Float64}}(undef, batch_size)

  if scale isa Vector
      if length(scale) != batch_size
          error("Scale vector length must match batch size")
      end
      for b in 1:batch_size
          s = scale[b]
          st = s isa Float64 ? (s, s, s) : s
          push!(new_sizes, Tuple(round.(Int, old_size .* st)))

          M = Matrix{Float64}(I, 4, 4)
          M[1,1] = st[1]
          M[2,2] = st[2]
          M[3,3] = st[3]
          matrices[b] = M
      end
  else
      # Shared
      scale_tuple = scale isa Float64 ? (scale, scale, scale) : scale
      ns = Tuple(round.(Int, old_size .* scale_tuple))
      push!(new_sizes, ns) # Just check one

      M = Matrix{Float64}(I, 4, 4)
      M[1,1] = scale_tuple[1]
      M[2,2] = scale_tuple[2]
      M[3,3] = scale_tuple[3]
      for b in 1:batch_size
         matrices[b] = M
      end
  end

  # Check output sizes consistency
  target_size = scale isa Vector ? new_sizes[1] : new_sizes[1]

  if scale isa Vector
      if !all(x -> x == target_size, new_sizes)
          error("Batched scaling with different factors must result in the same output voxel dimensions. Got: $new_sizes")
      end
  end

  return affine_transform_mi(image, matrices, Interpolator; output_size=target_size)
end

function translate_mi(im::BatchedMedImage, translate_by::Union{Int64, Vector{Int64}}, translate_in_axis::Int64, Interpolator::Interpolator_enum)::BatchedMedImage
  # Translate changes origin. Voxel data stays same (no interpolation needed for integer translation if we just move origin).
  # But existing translate_mi implementation:
  # origin_val = im.origin[translate_in_axis] + translate_by * im.spacing[translate_in_axis]
  # It updates spatial metadata only.

  batch_size = size(im.voxel_data, 4)
  new_im = deepcopy(im)

  for b in 1:batch_size
      t_by = (translate_by isa Vector) ? translate_by[b] : translate_by

      origin_val = new_im.origin[b][translate_in_axis] + t_by * new_im.spacing[b][translate_in_axis]
      translated_origin = ntuple(i -> i == translate_in_axis ? origin_val : new_im.origin[b][i], 3)
      new_im.origin[b] = translated_origin
  end
  return new_im
end

function crop_mi(im::BatchedMedImage, crop_beg::Union{Tuple{Int64,Int64,Int64}, Vector{Tuple{Int64,Int64,Int64}}}, crop_size::Tuple{Int64,Int64,Int64}, Interpolator::Interpolator_enum)::BatchedMedImage
    # Output size is fixed by crop_size
    batch_size = size(im.voxel_data, 4)

    # Check if we can use a view or need to copy. Since offsets might vary, we can't create a simple 4D view if offsets vary.
    # We must construct a new array.

    new_voxel_data = similar(im.voxel_data, crop_size[1], crop_size[2], crop_size[3], batch_size)
    new_origins = deepcopy(im.origin)

    for b in 1:batch_size
        cb = (crop_beg isa Vector) ? crop_beg[b] : crop_beg

        julia_beg = cb .+ 1
        # Extract slice
        # Verify bounds?
        # Assuming valid
        slice = view(im.voxel_data,
            julia_beg[1]:(julia_beg[1]+crop_size[1]-1),
            julia_beg[2]:(julia_beg[2]+crop_size[2]-1),
            julia_beg[3]:(julia_beg[3]+crop_size[3]-1),
            b)

        new_voxel_data[:, :, :, b] .= slice

        dir_diag = (im.direction[b][1], im.direction[b][5], im.direction[b][9])
        new_origins[b] = im.origin[b] .+ (im.spacing[b] .* cb .* dir_diag)
    end

    # Can't use update_voxel_data simply because origin changed too
    # But update_voxel_data handles metadata copy.
    # We can use update_voxel_data and then set origin.
    # Or just construct new struct.
    # update_voxel_data is differentiable for voxel_data

    new_im = update_voxel_data(im, new_voxel_data)
    new_im.origin = new_origins # Mutation! This might break Zygote if origin tracking is needed?
    # Zygote allows mutation of non-tracked fields? Origin is tracked in MedImage?
    # MedImage fields are typed.
    # To be Zygote safe, we should construct fully.
    # But for now let's fix the deepcopy.
    return new_im
end

function pad_mi(im::BatchedMedImage, pad_beg::Tuple{Int64,Int64,Int64}, pad_end::Tuple{Int64,Int64,Int64}, pad_val, Interpolator::Interpolator_enum)::BatchedMedImage
  # Pad params shared
  batch_size = size(im.voxel_data, 4)

  # Process each channel? Or treat batch dim as channel?
  # pad_dim only handles 3D array?
  # function pad_dim(arr, dim, l, r, val)
  # It takes arr.
  # We can pad 4D array in dims 1, 2, 3.

  data = im.voxel_data
  data = pad_dim(data, 1, pad_beg[1], pad_end[1], pad_val)
  data = pad_dim(data, 2, pad_beg[2], pad_end[2], pad_val)
  data = pad_dim(data, 3, pad_beg[3], pad_end[3], pad_val)

  new_origins = deepcopy(im.origin)
  for b in 1:batch_size
      dir_diag = (im.direction[b][1], im.direction[b][5], im.direction[b][9])
      new_origins[b] = im.origin[b] .- (im.spacing[b] .* pad_beg .* dir_diag)
  end

  new_im = update_voxel_data(im, data)
  new_im.origin = new_origins
  return new_im
end

# --- Affine Transformation ---

"""
    create_affine_matrix(translation=(0,0,0), rotation=(0,0,0), scale=(1,1,1), shear=(0,0,0))

Creates a 4x4 homogeneous affine transformation matrix.
Order of operations: Scale * Shear * Rotation * Translation
(Points are transformed as M * p)
Rotation angles are in degrees.
"""
function create_affine_matrix(; translation=(0.0,0.0,0.0), rotation=(0.0,0.0,0.0), scale=(1.0,1.0,1.0), shear=(0.0,0.0,0.0))
    # Translation Matrix
    T = Matrix{Float64}(I, 4, 4)
    T[1:3, 4] .= translation

    # Rotation Matrix (Z * Y * X)
    rx, ry, rz = deg2rad.(rotation)
    Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)]
    Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)]
    Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1]
    R_mat = Rz * Ry * Rx

    R = Matrix{Float64}(I, 4, 4)
    R[1:3, 1:3] = R_mat

    # Scale Matrix
    S = Matrix{Float64}(I, 4, 4)
    S[1,1] = scale[1]
    S[2,2] = scale[2]
    S[3,3] = scale[3]

    # Shear Matrix (simplified for now, xy, xz, yz, etc.)
    # Shear can be complex. Typically:
    # [1 xy xz 0]
    # [yx 1 yz 0]
    # [zx zy 1 0]
    # [0 0 0 1]
    # Here we assume shear=(xy, xz, yz) for upper triangular part, or symmetric?
    # Let's support 3 shear components: xy (shear x w.r.t y), xz, yz
    Sh = Matrix{Float64}(I, 4, 4)
    Sh[1,2] = shear[1]
    Sh[1,3] = shear[2]
    Sh[2,3] = shear[3]

    # Combine: Translation * Rotation * Shear * Scale
    # Applied right to left on point column vector: p' = T * R * Sh * S * p
    return T * R * Sh * S
end

function compose_affine_matrices(matrices...)
    result = Matrix{Float64}(I, 4, 4)
    for m in matrices
        result = m * result # Pre-multiply? Or post? Composition: M2 * M1 * p
        # If input order is (M1, M2), and we want M2(M1(p)), then result = M2 * M1
        # The loop does: M * result. So if we pass (M2, M1), we get M1 * M2?
        # Wait. result starts as I.
        # 1. result = m1 * I = m1
        # 2. result = m2 * m1
        # So matrices should be passed in order of application from right to left?
        # Typically compose(A, B) means A(B(x)) -> A * B.
        # So we update result = m * result.
    end
    return result
end

"""
    affine_transform_mi(image::BatchedMedImage, affine_matrix, Interpolator::Interpolator_enum; output_size=nothing)

Applies an affine transformation to a batch of images.
`affine_matrix` can be a single 4x4 matrix (shared) or a Vector of 4x4 matrices (unique per batch).
Transform is applied in index space relative to the image center.
"""
function affine_transform_mi(image::BatchedMedImage, affine_matrix::Union{Matrix{Float64}, Vector{Matrix{Float64}}}, Interpolator::Interpolator_enum; output_size=nothing)::BatchedMedImage
    batch_size = size(image.voxel_data, 4)

    spatial_size = (output_size === nothing) ? size(image.voxel_data)[1:3] : output_size

    # 1. Prepare Inverse Affine Matrices (CPU)
    # We invert here because we map Output -> Input
    matrices_inv = zeros(Float32, 4, 4, batch_size)

    for b in 1:batch_size
        M = (affine_matrix isa Vector) ? affine_matrix[b] : affine_matrix
        M_inv = try
            inv(M)
        catch
            error("Affine matrix is singular and cannot be inverted.")
        end
        matrices_inv[:, :, b] = M_inv
    end

    # 2. Determine backend and move matrices
    backend = KernelAbstractions.get_backend(image.voxel_data)

    if !(backend isa KernelAbstractions.CPU)
         # Assume using CUDA or similar
         # We need to construct backend-specific array
         # Simple generic way:
         # Use KernelAbstractions.zeros to init and copy? Or just convert if we knew the type.
         # For now, we rely on the fact that if image.voxel_data is CuArray, we can use CuArray(matrices_inv).
         # But we want to be generic.
         # Utils.cast_to_array_b_type? No.
         # We can try to match the array type container.
         # Or simply use KernelAbstractions.allocate?

         # Note: `is_cuda_array` is available.
         if is_cuda_array(image.voxel_data)
             # This requires CUDA.jl to be loaded if we call CuArray explicitly,
             # but Utils imports it.
             # Better to use a backend-agnostic way or rely on Utils helper?
             # For now, let's assume if it is CUDA array, we can make CuArray.
             # However, Basic_transformations doesn't depend on CUDA explicitly.
             # We can define a helper in Utils to "move_to_device(data, reference_array)".
             # Or just use `similar` and copyto!

             m_gpu = similar(image.voxel_data, Float32, 4, 4, batch_size)
             copyto!(m_gpu, matrices_inv)
             matrices_inv = m_gpu
         end
    end

    # 3. Generate Coords (GPU friendly)
    points_to_interpolate = generate_affine_coords(spatial_size, matrices_inv, backend)

    # 4. Interpolate
    spacing_arg = [ (1.0, 1.0, 1.0) for _ in 1:batch_size ]

    resampled_flat = interpolate_my(points_to_interpolate, image.voxel_data, spacing_arg, Interpolator, false, 0.0, true)

    new_data = reshape(resampled_flat, spatial_size[1], spatial_size[2], spatial_size[3], batch_size)

    return update_voxel_data(image, new_data)
end

end#Basic_transformations
