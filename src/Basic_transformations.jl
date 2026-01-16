module Basic_transformations

using CoordinateTransformations, Interpolations, StaticArrays, LinearAlgebra, Rotations, Dictionaries
using LinearAlgebra
using ImageTransformations
using ..MedImage_data_struct
using ..MedImage_data_struct: Nearest_neighbour_en, Linear_en, B_spline_en
using ..Load_and_save: update_voxel_data, update_voxel_and_spatial_data
using ..Utils: interpolate_my

export rotate_mi, crop_mi, pad_mi, translate_mi, scale_mi, computeIndexToPhysicalPointMatrices_Julia, transformIndexToPhysicalPoint_Julia, get_voxel_center_Julia, get_real_center_Julia, Rodrigues_rotation_matrix, crop_image_around_center

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

function get_real_center_Julia(im::MedImage)::Tuple{Vararg{Float64}}
  real_size = transformIndexToPhysicalPoint_Julia(im, size(im.voxel_data))
  real_origin = transformIndexToPhysicalPoint_Julia(im, (0, 0, 0))
  return Tuple((real_size .+ real_origin) ./ 2)
end


function Rodrigues_rotation_matrix(image::MedImage, axis::Int, angle::Float64)::Matrix{Float64}
  img_direction = image.direction
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


function rotate_mi(image::MedImage, axis::Int, angle::Float64, Interpolator::Interpolator_enum, crop::Bool=true)::MedImage
  R = Rodrigues_rotation_matrix(image, axis, angle)
  v_center = collect(get_voxel_center_Julia(image.voxel_data))

  rotation_transformation = LinearMap(RotXYZ(R))
  translation = Translation(v_center...)
  transkation_center = Translation(-v_center...)
  combined_transformation = translation ∘ rotation_transformation ∘ transkation_center

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
   left_block = fill(convert(eltype(arr), val), left_shape)

   right_shape = ntuple(i -> i == dim ? r : sz[i], length(sz))
   right_block = fill(convert(eltype(arr), val), right_shape)

   cat(left_block, arr, right_block; dims=dim)
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


function scale_mi(im::MedImage, scale::Union{Float64, Tuple{Float64,Float64,Float64}}, Interpolator::Interpolator_enum)::MedImage

  scale_tuple = scale isa Float64 ? (scale, scale, scale) : scale
  old_size = size(im.voxel_data)
  new_size = Tuple(round.(Int, old_size .* scale_tuple))

  new_data = if Interpolator == Nearest_neighbour_en
    imresize(im.voxel_data, new_size, method=Constant())
  elseif Interpolator == Linear_en
    imresize(im.voxel_data, new_size, method=Linear())
  else
    imresize(im.voxel_data, new_size, method=Linear())
  end

  new_im = update_voxel_and_spatial_data(im, new_data, im.origin, im.spacing, im.direction)

  return new_im
end

end#Basic_transformations
