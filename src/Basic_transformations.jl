module Basic_transformations

using CoordinateTransformations, Interpolations, StaticArrays, LinearAlgebra, Rotations, Dictionaries
using LinearAlgebra
using ImageTransformations
using ..MedImage_data_struct
using ..Load_and_save: update_voxel_data
export rotate_mi, crop_mi, pad_mi, translate_mi, scale_mi, computeIndexToPhysicalPointMatrices_Julia, transformIndexToPhysicalPoint_Julia, get_voxel_center_Julia, get_real_center_Julia, Rodrigues_rotation_matrix, crop_image_around_center

"""
given a MedImage object will rotate it by angle (angle) around axis (rotate_axis)
the center of rotation is set to be center of the image
It modifies both pixel array and not metadata
we are setting Interpolator by using Interpolator enum
return the rotated MedImage object
"""
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
  #physicalPointToIndex = inv(indexToPhysicalPoint)

  return indexToPhysicalPoint #, physicalPointToIndex
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
  #=
  Rotarion matrix using Rodrigues' rotation formula
  !"As it currently stands, it only supports 3D!
  =#
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
  # print(" \n ccccrrrrrrrrrr orig $(size(image)) cropped_image $(size(cropped_image)) start_z $(start_z) end_z $(end_z) start_y $(start_y) end_y $(end_y) start_x $(start_x) end_x $(end_x) \n")

  return cropped_image
end


function rotate_mi(image::MedImage, axis::Int, angle::Float64, Interpolator::Interpolator_enum, crop::Bool=true)::MedImage
  # Compute the rotation matrix

  R = Rodrigues_rotation_matrix(image, axis, angle)
  v_center = collect(get_voxel_center_Julia(image.voxel_data))
  img = convert(Array{Float64,3}, image.voxel_data)
  rotation_transformation = LinearMap(RotXYZ(R))
  translation = Translation(v_center...)
  transkation_center = Translation(-v_center...)
  combined_transformation = translation ∘ rotation_transformation ∘ transkation_center
  resampled_image = collect(warp(img, combined_transformation, Interpolations.Linear()))
  if crop
    new_center = get_voxel_center_Julia(resampled_image)
    resampled_image = crop_image_around_center(resampled_image, size(img), map(x -> round(Int, x), new_center))
    image = update_voxel_data(image, resampled_image)
  end

  return image
end





"""
given a MedImage object and a Tuples that contains the location of the begining of the crop (crop_beg) and the size of the crop (crop_size) crops image
It modifies both pixel array and metadata
we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used)
return the cropped MedImage object
"""
function crop_mi(im::MedImage, crop_beg::Tuple{Int64,Int64,Int64}, crop_size::Tuple{Int64,Int64,Int64}, Interpolator::Interpolator_enum)::MedImage

  # Create a view of the original voxel_data array with the specified crop
  cropped_voxel_data = @view im.voxel_data[crop_beg[1]:(crop_beg[1]+crop_size[1]-1), crop_beg[2]:(crop_beg[2]+crop_size[2]-1), crop_beg[3]:(crop_beg[3]+crop_size[3]-1)]

  # Adjust the origin according to the crop beginning coordinates
  cropped_origin = im.origin .+ (im.spacing .* crop_beg)

  # Create a new MedImage with the cropped voxel_data and adjusted origin
  cropped_im = update_voxel_and_spatial_data(im, cropped_voxel_data, cropped_origin, im.spacing, im.direction)

  return cropped_im

end#crop_mi


"""
given a MedImage object and a Tuples that contains the information on how many voxels to add in each axis (pad_beg) and on the end of the axis (pad_end)
we are performing padding by adding voxels with value pad_val
It modifies both pixel array and metadata
we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used)
return the cropped MedImage object
"""
function pad_mi(im::MedImage, pad_beg::Tuple{Int64,Int64,Int64}, pad_end::Tuple{Int64,Int64,Int64}, pad_val, Interpolator::Interpolator_enum)::MedImage

  # Create padding arrays for the beginning and end of each axis
  pad_beg_array = fill(pad_val, (pad_beg[1], im.voxel_data.size[2], im.voxel_data.size[3]))
  pad_end_array = fill(pad_val, (pad_end[1], im.voxel_data.size[2], im.voxel_data.size[3]))

  # Concatenate the padding arrays with the original voxel_data array
  padded_voxel_data = cat(pad_beg_array, im.voxel_data, pad_end_array, dims=1)

  # Adjust the origin according to the padding beginning coordinates
  padded_origin = im.origin .- (im.spacing .* pad_beg)

  # Create a new MedImage with the padded voxel_data and adjusted origin
  padded_im = update_voxel_and_spatial_data(im, padded_voxel_data, padded_origin, im.spacing, im.direction)
  return padded_im
end#pad_mi




"""
given a MedImage object translation value (translate_by) and axis (translate_in_axis) in witch to translate the image return translated image
It is diffrent from pad by the fact that it changes only the metadata of the image do not influence pixel array
we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used)
return the translated MedImage object
"""
function translate_mi(im::MedImage, translate_by::Int64, translate_in_axis::Int64, Interpolator::Interpolator_enum)::MedImage

  # Create a copy of the origin
  translated_origin = copy(im.origin)

  # Modify the origin according to the translation value and axis
  translated_origin[translate_in_axis] += translate_by * im.spacing[translate_in_axis]

  # Create a new MedImage with the translated origin and the original voxel_data
  translated_im = update_voxel_and_spatial_data(im, im.voxel_data, translated_origin, im.spacing, im.direction)

  return translated_im

end#crop_mi


"""
given a MedImage object and a Tuple that contains the scaling values for each axis (x,y,z in order)

we are setting Interpolator by using Interpolator enum
return the scaled MedImage object
"""
function scale_mi(im::MedImage, scale::Tuple{Float64,Float64,Float64}, Interpolator::Interpolator_enum)::MedImage

  # Determine the interpolation method
  interp_method = nothing
  if interpolator == Nearest_neighbour_en
    interp_method = Nearest
  elseif interpolator == Linear_en
    interp_method = BSpline(Linear())
  elseif interpolator == B_spline_en
    interp_method = BSpline(Cubic(Flat(OnGrid())))
  end

  # Scale the image
  new_data = imresize(im.voxel_data, scale, method=interp_method)

  # Create a new MedImage with the scaled data and the same spatial metadata
  new_im = update_voxel_and_spatial_data(im, new_data, translated_origin, im.spacing, im.direction)

  return new_im

end#scale_mi




#=
Testa i old functions

old functions:
-----------------TransformIndexToPhysicalPoint--------------------------------------------------
# prostsze podejście - zwraca wyłacznie dla 3D - błędnie wylicza y
function transformIndexToPhysicalPoint_Julia_v1(im::MedImage, index::Tuple{Int, Int, Int})::Tuple{Float64, Float64, Float64}
  point = (
      im.origin[1] + index[1] * im.spacing[1],
      im.origin[2] + index[2] * im.spacing[2],
      im.origin[3] + index[3] * im.spacing[3]
  )
  return point
end



struct Rodrigues3DTransform
  center::Vector{Float64}
  rotation_matrix::Matrix{Float64} # Nowe pole dla macierzy rotacji
  translation::Vector{Float64}
end

function Rodrigues3DTransform(center=[0.0, 0.0, 0.0], rotation_matrix=Matrix{Float64}(I, 3, 3), translation=[0.0, 0.0, 0.0])
    new(center, rotation_matrix, translation)
end


function set_rotation_matrix!(transform::Rodrigues3DTransform, rotation_matrix::Matrix{Float64})
    transform.rotation_matrix = rotation_matrix
end

function set_center!(transform::Rodrigues3DTransform, new_center::Tuple{Vararg{Float64}})
  new_center= collect(new_center)
  transform.center = new_center
end

function Rodrigues_transform_point(transform::Rodrigues3DTransform, point::Vector{Float64})
    R = transform.rotation_matrix
    shifted_point = point - transform.center
    transformed_point = R * shifted_point + transform.center + transform.translation
    return transformed_point
end




Testry
-----------------Test PhysicalPoint 3 i 4 D--------------------------------------------------

function test_4D_To_PhysicalPoint(image_path::String)
  image_test1 = sitk.ReadImage(image_path)
  # Zakładam, że `get_spatial_metadata` zwraca odpowiedni obiekt MedImage.
  image_test2 = get_spatial_metadata(image_path)
  # Pobranie rozmiaru obrazu.
  size = image_test1.GetSize()
  errors = 0
  all_iter = 0
  all_pixels = size[1]*size[2]*size[3]*size[4]
  # Dostosowanie do przechowywania 4-wymiarowych indeksów i punktów.
  error_details = Vector{Tuple{Tuple{Int,Int,Int,Int}, Tuple{Float64, Float64, Float64, Float64}, Tuple{Float64, Float64, Float64, Float64}}}()



  # Iteracja przez wszystkie indeksy w oparciu o rozmiar obrazu.
  for t in 0:(size[4]-1)
      for x in 0:(size[1]-1)
          for y in 0:(size[2]-1)
              for z in 0:(size[3]-1)
                  index = (x, y, z, t)
                  test1 = image_test1.TransformIndexToPhysicalPoint(index)
                  indexToPhysicalPoint, _ = computeIndexToPhysicalPointMatrices_Julia(image_test2)
                  test2 = transformIndexToPhysicalPoint_Julia(image_test2, index, indexToPhysicalPoint)
                  all_iner += 1
                  if !all(isapprox(test1[i], test2[i], atol=1e-5) for i in 1:4)
                      errors += 1
                      push!(error_details, (index, test1, test2))
                  end
              end
          end
      end
  end

  println("Liczba błędów: $errors")
  println("All interations: $(all_iter)")
  println("All pixels: $(all_pixels)")
  if errors > 0
      println("Szczegóły błędów (pierwsze 10):")
      for detail in error_details[1:min(10, end)]
          println("Indeks: $(detail[1]), Test1: $(detail[2]), Test2: $(detail[3])")
      end
  end
end


test_4D_To_PhysicalPoint("C:\\MedImage\\MedImage.jl\\test_data\\filtered_func_data.nii.gz")

function test_3D_To_PhysicalPoint(image_path::String)
  image_test1 = sitk.ReadImage(image_path)
  image_test2 = get_spatial_metadata(image_path)
  size = image_test1.GetSize()
  errors = 0
  all_iter = 0
  all_pixels = size[1]*size[2]*size[3]
  error_details = Vector{Tuple{Tuple{Int,Int,Int}, Tuple{Float64, Float64, Float64}, Tuple{Float64, Float64, Float64}}}()
  for x in 0:(size[1]-1)
      for y in 0:(size[2]-1)
          for z in 0:(size[3]-1)
              index = (x, y, z)
              test1 = image_test1.TransformIndexToPhysicalPoint(index)
              indexToPhysicalPoint, _ = computeIndexToPhysicalPointMatrices_Julia(image)
              test2 = transformIndexToPhysicalPoint_Julia(image_test2, index, indexToPhysicalPoint)
              all_iner += 1
              if !all(isapprox(test1[i], test2[i], atol=1e-5) for i in 1:3)
                  errors += 1
                  push!(error_details, (index, test1, test2))
              end
          end
      end
  end

  println("Liczba błędów: $errors")
  println("All interations: $(all_iter)")
  println("All pixels: $(all_pixels)")
  if errors > 0
    println("Szczegóły błędów (pierwsze 10):")
    for detail in error_details[1:min(100, end)]
        println("Indeks: $(detail[1]), Test1: $(detail[2]), Test2: $(detail[3])")
        println("All interations: $all_iner")
    end
  end
end

test_3D_To_PhysicalPoint("C:\\MedImage\\MedImage.jl\\test_data\\volume-0.nii.gz")


=#
end#Basic_transformations
