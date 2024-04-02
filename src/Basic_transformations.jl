include("MedImage_data_struct.jl")
using ImageTransformations, CoordinateTransformations, Interpolations, StaticArrays, LinearAlgebra, Rotations, Dictionaries
"""
module implementing basic transformations on 3D images 
like translation, rotation, scaling, and cropping both input and output are MedImage objects
"""

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


function get_voxel_center_Julia(image::Array{T, 3})::Tuple{Vararg{Float64}} where T
  real_size = size(image)
  real_origin = (0,0,0)
  return Tuple((real_size .+ real_origin) ./ 2)
end

function get_real_center_Julia(im::MedImage)::Tuple{Vararg{Float64}}
  real_size = transformIndexToPhysicalPoint_Julia(im,size(im.voxel_data))
  real_origin = transformIndexToPhysicalPoint_Julia(im,(0,0,0))
  return Tuple((real_size .+ real_origin) ./ 2)
end


function Rodrigues_rotation_matrix(im::MedImage, axis::String, angle::Float64)::Matrix{Float64}
  im_direction = im.direction
  axis_angle = if axis == "X"
    (im_direction[9], im_direction[6], im_direction[3])
  elseif axis == "Y"
      (im_direction[8], im_direction[5], im_direction[2])
  elseif axis == "Z"
      (im_direction[7], im_direction[4], im_direction[1])
  end
  ux, uy, uz= axis_angle
  theta = deg2rad(angle)
  c = cos(theta)
  s = sin(theta)
  ci = 1.0 - c
  R = [ci * ux * ux + c   ci * ux * uy - uz * s ci * ux * uz + uy * s;
      ci * uy * ux + uz * s ci * uy * uy + c   ci * uy * uz - ux * s;
      ci * uz * ux - uy * s ci * uz * uy + ux * s ci * uz * uz + c]
  return R
end

function crop_image_around_center(image::Array{T, 3}, new_dims::Tuple{Int, Int, Int}, center::Tuple{Int, Int, Int}) where T
  start_z = max(1, center[1] - new_dims[1] ÷ 2)
  end_z = min(size(image, 1), start_z + new_dims[1] - 1)
  
  start_y = max(1, center[2] - new_dims[2] ÷ 2)
  end_y = min(size(image, 2), start_y + new_dims[2] - 1)

  start_x = max(1, center[3] - new_dims[3] ÷ 2)
  end_x = min(size(image, 3), start_x + new_dims[3] - 1)

  cropped_image = image[start_z:end_z, start_y:end_y, start_x:end_x]
  return cropped_image
end



function rotation_and_resample(image::MedImage, axis::String, angle::Float64, cropp::Bool=true)::MedImage
  R = Rodrigues_rotation_matrix(image, axis, angle)
  v_center = collect(get_voxel_center_Julia(image.voxel_data))
  img = convert(Array{Float64, 3}, image.voxel_data)
  rotation_transformation = LinearMap(RotXYZ(R))
  translation = Translation(v_center...)
  transkation_center = Translation(-v_center...)
  combined_transformation = translation ∘ rotation_transformation ∘ transkation_center
  resampled_image = collect(warp(img, combined_transformation, Interpolations.Linear()))
  if cropp
    resampled_image =crop_image_around_center(resampled_image, size(img), map(x -> round(Int, x), get_voxel_center_Julia(resampled_image)))
  end
  image = update_voxel_data(image, resampled_image)
  return image
end




#= Example of usage
image_3D=get_spatial_metadata("C:\\MedImage\\MedImage.jl\\test_data\\volume-0.nii.gz")
test = rotation_and_resample(image_3D, "X", 45.0)
create_nii_from_medimage(test, "C:\\MedImage\\MedImage.jl\\src\\test_rotacji\\nowy")
=#

"""
given a MedImage object and a Tuples that contains the location of the begining of the crop (crop_beg) and the size of the crop (crop_size) crops image
It modifies both pixel array and metadata
we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used)
return the cropped MedImage object 
"""
function crop_mi(im::Array{MedImage}, crop_beg::Tuple{Int64,Int64,Int64}, crop_size::Tuple{Int64,Int64,Int64}, Interpolator::Interpolator)::Array{MedImage}

  nothing

end#crop_mi    


"""
given a MedImage object and a Tuples that contains the information on how many voxels to add in each axis (pad_beg) and on the end of the axis (pad_end)
we are performing padding by adding voxels with value pad_val
It modifies both pixel array and metadata
we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used)
return the cropped MedImage object 
"""
function pad_mi(im::Array{MedImage}, pad_beg::Tuple{Int64,Int64,Int64}, pad_end::Tuple{Int64,Int64,Int64},pad_val, Interpolator::Interpolator)::Array{MedImage}

  nothing

end#pad_mi    




"""
given a MedImage object translation value (translate_by) and axis (translate_in_axis) in witch to translate the image return translated image
It is diffrent from pad by the fact that it changes only the metadata of the image do not influence pixel array
we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used)
return the translated MedImage object
"""
function translate_mi(im::Array{MedImage}, translate_by::Int64, translate_in_axis::Int64, Interpolator::Interpolator)::Array{MedImage}

  nothing

end#crop_mi    


"""
given a MedImage object and a Tuple that contains the scaling values for each axis (x,y,z in order)

we are setting Interpolator by using Interpolator enum
return the scaled MedImage object 
"""
function scale_mi(im::Array{MedImage}, scale::Tuple{Float64,Float64,Float64}, Interpolator::Interpolator)::Array{MedImage}

  nothing

end#scale_mi    




#=
Testry i stare funkcje

Stare funkcjie
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
-----------------Testy PhysicalPoint 3 i 4 D--------------------------------------------------

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
