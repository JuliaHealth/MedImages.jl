include("./Utils.jl")
include("./Spatial_metadata_change.jl")
include("./Resamle_to_.jl")
using Distributions, Random, Statistics, CUDA, KernelAbstractions

function augment_brightness(image::Union{MedImage,Array{Float32, 3}}, value::Float64, mode::String)::Array{Float32, 3}
    """
    Work in progres
    """
    im = union_check(image)

    if mode == "additive"
        im .+= value
    elseif mode == "multiplicative"
        im .*= value
    else
        error("Invalid mode. Choose 'additive' or 'multiplicative'.")
    end
    return im
end

function augment_contrast(image::Union{MedImage,Array{Float32, 3}}, factor::Float64)::Array{Float32, 3}
    """

    """
    im = union_check(image)

    mn = mean(im)
    im .= (im .- mn) .* factor .+ mn

    return im
end

function augment_gamma(image::Union{MedImage,Array{Float32, 3}}, gamma::Float64)::Array{Float32, 3}
    """
    Work in progres
    """
    im = union_check(image)
    min_val, max_val = extrema(im)

    normalized_data = (im .- min_val) ./ (max_val - min_val)
    transformed_data = normalized_data .^ gamma
    im .= (transformed_data .* (max_val - min_val)) .+ min_val

    return im
end

function augment_gaussian_noise(image::Union{MedImage,Array{Float32, 3}}, variance::Float64)::Array{Float32, 3}
    """
    Work in progres
    """
    im = union_check(image)

    noise = rand(Normal(0.0, variance), size(im))
    im .+= noise

    return im
end

function augment_rician_noise(image::Union{MedImage,Array{Float32, 3}}, variance::Float64)::Array{Float32, 3}
    """
    Work in progres
    """
    im = union_check(image)

    noise1 = rand(Normal(0.0, variance), size(im))
    noise2 = rand(Normal(0.0, variance), size(im))
    im .= sqrt.((im .+ noise1).^2 + noise2.^2) .* sign.(im)

    return im
end

function augment_mirror(image::Union{MedImage,Array{Float32, 3}}, axes=(1, 2, 3)::Tuple{Int, Int, Int})::Array{Float32, 3}
    """
    Work in progres
    """
    im = union_check(image)
    
    if 1 in axes
        im = im[end:-1:1, :, :]
    end
    if 2 in axes
        im = im[:, end:-1:1, :]
    end
    if 3 in axes
        im = im[:, :, end:-1:1]
    end

    return im
end

function augment_scaling_bySpacing(image::Union{MedImage,Array{Float32, 3}}, interpolator_enum)::Array{Float32, 3} #zmienić resample_to_spacing żeby działało na czystych arrayach
    """
    Work in progres
    """
    im = union_check(image)
    
    original_size = size(im) 
    new_spacing = image.spacing .* (1/scale_factor)
    image_scaled = resample_to_spacing(image, new_spacing, interpolator_enum) #tu się zapsuje
    new_size = size(image_scaled.voxel_data)

    if any(new_size .< original_size)
        pad_beg = Tuple((original_size .- new_size) .÷ 2)
        pad_end = Tuple(original_size .- new_size .- pad_beg)
        new_image = pad_mi_2(image_scaled, pad_beg, pad_end, extrapolate_median(im))
    elseif any(new_size .> original_size)
        crop_beg = Tuple((new_size .- original_size) .÷ 2)
        crop_size = original_size
        new_image = crop_mi(image_scaled, crop_beg, crop_size, interpolator_enum)
    end

    return new_image
end


function elastic_deformation3d(image::Union{MedImage,Array{Float32, 3}}, strength::Float64, interpolator_enum) where T
    """
    Work in progres
    """
    img = union_check(image)
    
    deformed_img = similar(img)

    # Inicjalizacja wektorów przesunięcia dla każdego punktu obrazu
    displacement_x = randn(size(img)...) * strength
    displacement_y = randn(size(img)...) * strength
    displacement_z = randn(size(img)...) * strength

    # Wybór interpolatora
    if interpolator_enum == :Nearest_neighbour_en
        itp = interpolate(img, BSpline(Constant()))
    elseif interpolator_enum == :Linear_en
        itp = interpolate(img, BSpline(Linear()))
    elseif interpolator_enum == :B_spline_en
        itp = interpolate(img, BSpline(Cubic(Line(OnGrid()))))
    else
        error("Nieznany typ interpolatora!")
    end

    # Wywołanie kernela
    kernel = elastic_deformation_kernel(CPU(), (8, 8, 8))(img, deformed_img, displacement_x, displacement_y, displacement_z, itp)

    return deformed_img
end

@kernel function elastic_deformation_kernel(img, deformed_img, displacement_x, displacement_y, displacement_z, size_x, size_y, size_z, itp)
    """
    Work in progres
    """
    x_global, y_global, z_global = @index(Global, Cartesian)  # Globalne indeksy kartezjańskie
    
    if 1 <= x_global <= size_x && 1 <= y_global <= size_y && 1 <= z_global <= size_z
        new_x = x_global + displacement_x[x_global, y_global, z_global]
        new_y = y_global + displacement_y[x_global, y_global, z_global]
        new_z = z_global + displacement_z[x_global, y_global, z_global]

        if 1 <= new_x <= size_x && 1 <= new_y <= size_y && 1 <= new_z <= size_z
            deformed_img[x_global, y_global, z_global] = itp([new_x, new_y, new_z])  # Użycie funkcji interpolacji
        else
            deformed_img[x_global, y_global, z_global] = 0  # Poza granicami obrazu, np. padding
        end
    end
end



function augment_gaussian_blur(image::Union{MedImage,Array{Float32, 3}}, sigma::Float64, kernel_size::Int, shape="3D")::Array{Float32, 3}
    """
    Work in progres
    """
    if shape == "3D"
        return apply_padded_convolution_3D_GPU(image, sigma, kernel_size)
    if shape == "2D"
        return apply_padded_convolution(image, sigma, kernel_size)
    end

end
function create_gaussian_kernel(sigma, kernel_size)
    kernel_range = floor(Int, kernel_size / 2)
    kernel = [exp(-((x^2 + y^2) / (2 * sigma^2))) for x in -kernel_range:kernel_range, y in -kernel_range:kernel_range]
    kernel ./= sum(kernel)  # Normalizacja jądra
    return kernel
end

@kernel function padded_convolution_kernel(result, im, kernel, pad_x, pad_y)
    """
    Work in progres
    """
    idx = @index(Global, Cartesian)
    img_x, img_y, img_z = size(im)
    kernel_x, kernel_y = size(kernel)

    x, y, z = idx[1], idx[2], idx[3]

    # Rozszerzony zakres indeksów, aby uwzględnić padding
    ix_start = max(1, x - pad_x)
    ix_end = min(img_x, x + pad_x)
    iy_start = max(1, y - pad_y)
    iy_end = min(img_y, y + pad_y)

    value = 0.0
    for ix = ix_start:ix_end
        for iy = iy_start:iy_end
            m = ix - x + pad_x 
            n = iy - y + pad_y 
            if m > 0 && m <= kernel_x && n > 0 && n <= kernel_y
                value += im[ix, iy, z] * kernel[m, n]
            end
        end
    end
    if x >= pad_x + 1 && x <= img_x - pad_x  && y >= pad_y + 1 && y <= img_y - pad_y
        result[x - pad_x, y - pad_y, z] = value  # Przesunięcie wyniku do odpowiedniej lokalizacji
    end
end

function apply_padded_convolution(image::Union{MedImage,Array{Float32, 3}}, sigma, kernel_size)::Array{Float32, 3}
    """
    Work in progres
    """
    im = union_check(image)
    kernel = create_gaussian_kernel(sigma, kernel_size)

    pad_x, pad_y = size(kernel) .÷ 2
    stretch = (pad_x, pad_x, pad_x)

    # Użyj własnej funkcji paddingu do dodania wartości brzegowych
    padded_im = pad_mi_4(im, stretch)

    img_x, img_y, img_z = size(padded_im)
    result = similar(padded_im)

    # Wywołanie kernela z odpowiednim zakresem i rozmiarem grupy roboczej
    ndrange = (img_x, img_y, img_z)
    workgroupsize = (8, 8, 1)  # Dostosuj do swojego sprzętu
    kernel_event = padded_convolution_kernel(CPU(), workgroupsize)(result, padded_im, kernel, pad_x, pad_y, ndrange = ndrange)
    synchronize(CPU())

    # Przytnij wynik do oryginalnego rozmiaru obrazu
    final_result = result[1:end-(pad_x*2), 1:end-(pad_x*2), 1:end-(pad_x*2)]
    return final_result
end

#Kernele z padowaniem w 3 wymiarach jednocześnie

function create_gaussian_kernel_3D(sigma, kernel_size)
    """
    Work in progres
    """
    kernel_range = floor(Int, kernel_size / 2)
    kernel = [exp(-((x^2 + y^2 + z^2) / (2 * sigma^2))) for x in -kernel_range:kernel_range, y in -kernel_range:kernel_range, z in -kernel_range:kernel_range]
    kernel ./= sum(kernel)
    return kernel
end

@kernel function padded_convolution_kernel_3D(result, im, kernel, pad_x, pad_y, pad_z)
    """
    Work in progres
    """
    idx = @index(Global, Cartesian)
    img_x, img_y, img_z = size(im)
    kernel_x, kernel_y, kernel_z = size(kernel)

    x, y, z = idx[1], idx[2], idx[3]

    # Rozszerzony zakres indeksów, aby uwzględnić padding
    ix_start = max(1, x - pad_x)
    ix_end = min(img_x, x + pad_x)
    iy_start = max(1, y - pad_y)
    iy_end = min(img_y, y + pad_y)
    iz_start = max(1, z - pad_z)
    iz_end = min(img_z, z + pad_z)

    value = 0.0
    for ix = ix_start:ix_end
        for iy = iy_start:iy_end
            for iz = iz_start:iz_end
                m = ix - x + pad_x + 1
                n = iy - y + pad_y + 1
                p = iz - z + pad_z + 1 
                if m > 0 && m <= kernel_x && n > 0 && n <= kernel_y && p > 0 && p <= kernel_z
                    value += im[ix, iy, iz] * kernel[m, n, p]
                end
            end
        end
    end
    if x >= pad_x + 1 && x <= img_x - pad_x && y >= pad_y + 1 && y <= img_y - pad_y && z >= pad_z + 1 && z <= img_z - pad_z
        result[x - pad_x, y - pad_y, z - pad_z] = value
    end
end

function apply_padded_convolution_3D_GPU(image::Union{MedImage,Array{Float32, 3}}, sigma, kernel_size)::Array{Float32, 3}
    """
    Work in progres
    """
    im = union_check(image)
    kernel = create_gaussian_kernel_3D(sigma, kernel_size)
    pad_x, pad_y, pad_z = size(kernel) .÷ 2
    padded_im = pad_mi_4(im, (pad_x, pad_y, pad_z))
    img_x, img_y, img_z = size(padded_im)
    result_gpu = CuArray(similar(padded_im))
    im_gpu = CuArray(im)
    
    # Wyznaczanie rozmiarów gridu i bloku
    dev = get_backend(im_gpu)
    workgroupsize = (16, 16, 16)
    ndrange = (img_x, img_y, img_z)

    # Uruchomienie kernela na GPU
    kernel_event = padded_convolution_kernel_3D(dev, workgroupsize)(result_gpu, padded_im, kernel_gpu, pad_x, pad_y, pad_z, ndrange = ndrange)
    # Zwracanie wyniku na CPU, jeśli potrzeba
    return Array(result_gpu)
end


function augment_simulate_low_resolution(image::Union{MedImage,Array{Float32, 3}}, blur_sigma::Float64, kernel_size::Int, downsample_scale::Float64)::Array{Float32, 3} #tutaj też jest syf z typami i utils
    """
    Work in progres
    """
    im = union_check(image)
    blurred_voxel_data = augment_gaussian_blur(im, blur_sigma, kernel_size)
    image_downsampled = augment_scaling_bySpacing(blurred_voxel_data, downsample_scale)
    image_upsampled = augment_scaling_bySpacing(image_downsampled, 1/downsample_scale)

    return image_upsampled
end



################### Temporary place for utils functions ###################

function union_check(image::Union{MedImage, Array{Float32, 3}})
    """
    Work in progres
    """
    if image isa MedImage
        im = copy(image.voxel_data)
    elseif image isa Array{Float32, 3}
        im = copy(image)
    else
        error("Invalid input type. Choose MedImage or Array{Float32, 3}.")
    end
    return im
end

function extrapolate_median(im::Array{Float32, 3})
    """
    Work in progres
    """
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
    value_to_extrapolate=median(corners)
    return value_to_extrapolate
end

function pad_mi_2(im::MedImage, pad_beg::Tuple{Int64, Int64, Int64}, pad_end::Tuple{Int64, Int64, Int64}, pad_val)
    """
    Work in progres
    """
    pad_beg_x = fill(pad_val, (pad_beg[1], size(im.voxel_data, 2), size(im.voxel_data, 3)))
    pad_end_x = fill(pad_val, (pad_end[1], size(im.voxel_data, 2), size(im.voxel_data, 3)))
    padded_voxel_data = vcat(pad_beg_x, im.voxel_data, pad_end_x)
    
    pad_beg_y = fill(pad_val, (size(padded_voxel_data, 1), pad_beg[2], size(im.voxel_data, 3)))
    pad_end_y = fill(pad_val, (size(padded_voxel_data, 1), pad_end[2], size(im.voxel_data, 3)))
    padded_voxel_data = hcat(pad_beg_y, padded_voxel_data, pad_end_y)


    pad_beg_z = fill(pad_val, (size(padded_voxel_data, 1), size(padded_voxel_data, 2), pad_beg[3]))
    pad_end_z = fill(pad_val, (size(padded_voxel_data, 1), size(padded_voxel_data, 2), pad_end[3]))
    padded_voxel_data = cat(pad_beg_z, padded_voxel_data, pad_end_z, dims=3)

    padded_origin = im.origin .- (im.spacing .* pad_beg)  # Adjust the origin

    padded_im = update_voxel_and_spatial_data(im, padded_voxel_data, padded_origin, im.spacing, im.direction)
    return padded_im
end

function pad_mi_4(im::Array{Float32, 3}, strech::Tuple{Int64, Int64, Int64})
    """
    Work in progres
    """
    im = union_check(im)

    pad_beg_x = im[1:1, :, :] 
    pad_end_x = im[end:end, :, :] 
    padded_voxel_data = im
    for i in 1:strech[1]
        padded_voxel_data = cat(pad_beg_x, padded_voxel_data, pad_end_x, dims=1)
    end
    pad_beg_y = padded_voxel_data[:, 1:1, :]  
    pad_end_y = padded_voxel_data[:, end:end, :]  
    for i in 1:strech[2]
        padded_voxel_data = cat(pad_beg_y, padded_voxel_data, pad_end_y, dims=2)
    end
    pad_beg_z = padded_voxel_data[:, :, 1:1] 
    pad_end_z = padded_voxel_data[:, :, end:end]
    for i in 1:strech[3]
        padded_voxel_data = cat(pad_beg_z, padded_voxel_data, pad_end_z, dims=3)
    end
    return padded_voxel_data
end

