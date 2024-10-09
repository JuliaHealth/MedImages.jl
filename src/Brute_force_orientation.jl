module Brute_force_orientation
using ..Orientation_dicts, ..MedImage_data_struct

using Interpolations
using Combinatorics
using JLD, PyCall

export orientation_enum_to_string
export change_image_orientation
export brute_force_find_perm_rev
export brute_force_find_perm_spacing
export establish_orginn_transformation
export brute_force_find_from_sitk_single
export brute_force_find_from_sitk
export get_orientations_vectors

orientation_enum_to_string = Dict(
    # ORIENTATION_RIP=>"RIP",
    # ORIENTATION_LIP=>"LIP",
    # ORIENTATION_RSP=>"RSP",
    # ORIENTATION_LSP=>"LSP",
    # ORIENTATION_RIA=>"RIA",
    # ORIENTATION_LIA=>"LIA",
    # ORIENTATION_RSA=>"RSA",
    # ORIENTATION_LSA=>"LSA",
    # ORIENTATION_IRP=>"IRP",
    # ORIENTATION_ILP=>"ILP",
    # ORIENTATION_SRP=>"SRP",
    # ORIENTATION_SLP=>"SLP",
    # ORIENTATION_IRA=>"IRA",
    # ORIENTATION_ILA=>"ILA",
    # ORIENTATION_SRA=>"SRA",
    # ORIENTATION_SLA=>"SLA",
    MedImage_data_struct.ORIENTATION_RPI => "RPI",
    MedImage_data_struct.ORIENTATION_LPI => "LPI",
    MedImage_data_struct.ORIENTATION_RAI => "RAI",
    MedImage_data_struct.ORIENTATION_LAI => "LAI",
    MedImage_data_struct.ORIENTATION_RPS => "RPS",
    MedImage_data_struct.ORIENTATION_LPS => "LPS",
    MedImage_data_struct.ORIENTATION_RAS => "RAS",
    MedImage_data_struct.ORIENTATION_LAS => "LAS",
    # ORIENTATION_PRI=>"PRI",
    # ORIENTATION_PLI=>"PLI",
    # ORIENTATION_ARI=>"ARI",
    # ORIENTATION_ALI=>"ALI",
    # ORIENTATION_PRS=>"PRS",
    # ORIENTATION_PLS=>"PLS",
    # ORIENTATION_ARS=>"ARS",
    # ORIENTATION_ALS=>"ALS",
    # ORIENTATION_IPR=>"IPR",
    # ORIENTATION_SPR=>"SPR",
    # ORIENTATION_IAR=>"IAR",
    # ORIENTATION_SAR=>"SAR",
    # ORIENTATION_IPL=>"IPL",
    # ORIENTATION_SPL=>"SPL",
    # ORIENTATION_IAL=>"IAL",
    # ORIENTATION_SAL=>"SAL",
    # ORIENTATION_PIR=>"PIR",
    # ORIENTATION_PSR=>"PSR",
    # ORIENTATION_AIR=>"AIR",
    # ORIENTATION_ASR=>"ASR",
    # ORIENTATION_PIL=>"PIL",
    # ORIENTATION_PSL=>"PSL",
    # ORIENTATION_AIL=>"AIL",
    # ORIENTATION_ASL=>"ASL"
)

string_to_orientation_enum = Dict(value => key for (key, value) in orientation_enum_to_string)





function change_image_orientation(path_nifti, orientation)
    sitk = pyimport("SimpleITK")

    # Read the image
    image = sitk.ReadImage(path_nifti)

    # Create a DICOMOrientImageFilter
    orient_filter = sitk.DICOMOrientImageFilter()

    # Convert the orientation enum to a string using the dictionary
    orientation_str = orientation_enum_to_string[orientation]
    # println("Setting orintation to : ", orientation_str)

    # Set the desired orientation (ensure it's a string)
    orient_filter.SetDesiredCoordinateOrientation(orientation_str)

    # Apply the filter to the image
    oriented_image = orient_filter.Execute(image)

    # Verify the orientation change
    # println("Original Orientation: ", image.GetDirection())
    # println("New Orientation: ", oriented_image.GetDirection())

    # Write the oriented image back to the file
    sitk.WriteImage(oriented_image, path_nifti)
end


"""
force solution get all direction combinetions put it in sitk and try all possible ways to permute and reverse axis to get the same result as sitk then save the result in json or sth and use; do the same with the origin
Additionally in similar manner save all directions in a form of a vector and associate it with 3 letter codes
"""
function brute_force_find_perm_rev(sitk_image1_arr, sitk_image2_arr)
    sitk = pyimport("SimpleITK")
    comb = collect(combinations([1, 2, 3]))
    push!(comb, [])

    perm = collect(permutations([1, 2, 3]))
    perm = filter(x -> x != [1, 2, 3], perm)
    push!(perm, [])
    for p in perm
        for c in comb
            if (length(p) > 0)
                curr = permutedims(sitk_image2_arr, (p[1], p[2], p[3]))
            else
                curr = sitk_image2_arr
            end

            if (length(c) == 1)
                curr = reverse(curr; dims=c[1])
            elseif (length(c) > 1)
                curr = reverse(curr; dims=Tuple(c))
            end
            if (curr == sitk_image1_arr)
                # println("Found")
                # println(p)
                # println(c)
                return (p, c)
            end

        end
    end

end


function brute_force_find_perm_spacing(spacing_source, spacing_target)
    sitk = pyimport("SimpleITK")
    comb = collect(combinations([1, 2, 3]))
    push!(comb, [])

    perm = collect(permutations([1, 2, 3]))
    # perm = filter(x -> x != [1,2,3], perm)
    # push!(perm,[])
    for p in perm
        if (length(p) > 0)
            curr = [spacing_source[p[1]], spacing_source[p[2]], spacing_source[p[3]]]
        else
            curr = spacing_source
        end

        if (curr == spacing_target)
            # println("Found spacing perm")
            # println(p)
            # println(c)
            return p
        end

    end

end



function establish_orginn_transformation(medim, sitk_image2, new_orientation, perm_main, reverse_axes, spacing_transforms)
    sitk = pyimport("SimpleITK")
    sizz = size(medim.voxel_data)
    origin1 = medim.origin
    origin2 = sitk_image2.GetOrigin()
    spacing1 = medim.spacing
    spacing2 = sitk_image2.GetSpacing()
    perm = collect(permutations([1, 2, 3]))


    res_list = []

    #spac_axis,sizz_axis,prim_origin_axis,op_sign
    spac_axises = [1, 2, 3]
    sizz_axises = [3, 2, 1]
    prim_origin_axises = [3, 2, 1]
    op_signs = [0, 1, -1]

    for sp in spac_axises
        for sz in sizz_axises
            for po in prim_origin_axises
                for op in op_signs
                    res_list = push!(res_list, [sp, sz, po, op])
                end
            end
        end
    end
    res_list_fin = []
    for i in res_list, j in res_list, k in res_list
        res_list_fin = push!(res_list_fin, [i, j, k])
    end


    for res in res_list_fin
        reorient_operation = (perm_main, reverse_axes, res, spacing_transforms)
        re_im = change_orientation_main(medim, new_orientation, reorient_operation)
        # print("\n tttttttttttt $(re_im.origin)  origin2 $(origin2)\n ")

        if (isapprox(collect(re_im.origin), collect(origin2); atol=0.1))
            print("ffffffffffffff")
            return res
        end
    end



    # print("\n rrrrres $(res) \n")
    # if(isapprox(p,collect(origin2); atol=0.1) )
    #     println("Found origin transformation $(i)")

    # # print("jjjj $(opts) origin2 $(origin2) ")
    # if(length(res[1])==0 || length(res[2])==0 || length(res[3])==0)
    println("\n Not found sizz $(collect(sizz).*collect(spacing1)) diff $(collect(origin1)-collect(origin2)) origin1 $(origin1) origin2 $(origin2) spacing1 $(spacing1) spacing2 $(spacing2) \n")

    diff = collect(origin1) - collect(origin2)
    # print("\n ooooo origin1 $(origin1) origin2 $(origin2) spacing1 $(spacing1) spacing2 $(spacing2) diff $(diff)  sizzz $(sizzz) \n")
    # end
    # return res
end



function brute_force_find_from_sitk_single(path_nifti, or_enum_1::Orientation_code, or_enum_2::Orientation_code)
    sitk = pyimport("SimpleITK")
    or_enum_1_str = orientation_enum_to_string[or_enum_1]
    or_enum_2_str = orientation_enum_to_string[or_enum_2]
    sitk_image1 = change_image_orientation(path_nifti, or_enum_1_str)
    sitk_image2 = change_image_orientation(path_nifti, or_enum_2_str)

    sitk_image1_arr = sitk.GetArrayFromImage(sitk_image1)
    sitk_image2_arr = sitk.GetArrayFromImage(sitk_image2)

    sitk_image1_arr = permutedims(sitk_image1_arr, (3, 2, 1))
    sitk_image2_arr = permutedims(sitk_image2_arr, (3, 2, 1))

    #find what permutation of axis and reversing is needed to get the same result as in sitk
    p, c = brute_force_find_perm_rev(sitk_image1_arr, sitk_image2_arr)

    p_spac = brute_force_find_perm_spacing(collect(sitk_image1.GetSpacing()), collect(sitk_image2.GetSpacing()))
    path_nifti_temp = "/home/jm/projects_new/MedImage.jl/test_data/synthethic_small_temp.nii.gz"
    sitk.WriteImage(sitk_image1, path_nifti_temp)
    med_im = load_image(path_nifti_temp)

    # get idea how to get ransformed origin
    origin_perm = establish_orginn_transformation(med_im, sitk_image2, or_enum_2, p, c, p_spac)
    return ((or_enum_1, or_enum_2), (p, c, origin_perm, p_spac))

end



function brute_force_find_from_sitk(path_nifti)
    sitk = pyimport("SimpleITK")
    opts = []
    for value_out in collect(instances(Orientation_code))
        for value_in in collect(instances(Orientation_code))
            opts = push!(opts, (value_out, value_in))
        end
    end

    return map(el -> brute_force_find_from_sitk_single(path_nifti, el[1], el[2]), opts)
end

function get_orientations_vectors(path_nifti)
    opts = []

    for orientation in collect(instances(Orientation_code))
        opts = push!(opts, (orientation, change_image_orientation(path_nifti, orientation_enum_to_string[orientation]).GetDirection()))
    end
    return opts
end

# path_nifti = "/home/jm/projects_new/MedImage.jl/test_data/volume-0.nii.gz"
# path_nifti = "/home/jm/projects_new/MedImage.jl/test_data/synthethic_small.nii.gz"

# # im=sitk.ReadImage("/home/jm/projects_new/MedImage.jl/test_data/volume-0.nii.gz")
# im=sitk.GetImageFromArray(rand(8,10,16))
# # im.SetSpacing([1.0,2.0,3.0])
# # im.SetOrigin((5.0,7.0,11.0))
# im.SetSpacing([1.06,2.055,3.13])
# im.SetOrigin((1.012,-2.2130,3.041))
# im.SetDirection([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
# sitk.WriteImage(im,"/home/jm/projects_new/MedImage.jl/test_data/synthethic_small.nii.gz")


# oo=get_orientations_vectors(path_nifti)
# oo
# dict_curr=Dict(oo)


# save("/home/jakubmitura/projects/MedImage.jl/test_data/dict_code_to_vector.jld", "dict_code_to_vector", dict_curr)
# loaded_dict = load("/home/jakubmitura/projects/MedImage.jl/test_data/my_dict.jld")

# for el in collect(loaded_dict)
#     println(",$(el)")
# end


# dict_enum_to_number=Dict(
#     ORIENTATION_ASR => (0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
#     ,ORIENTATION_AIL => (0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
#     ,ORIENTATION_PSR => (0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
#     ,ORIENTATION_ASL => (0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
#     ,ORIENTATION_RAS => (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
#     ,ORIENTATION_AIR => (0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
#     ,ORIENTATION_LIP => (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)
#     ,ORIENTATION_LAS => (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
#     ,ORIENTATION_SPL => (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
#     ,ORIENTATION_PLS => (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
#     ,ORIENTATION_ARI => (0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
#     ,ORIENTATION_LPI => (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
#     ,ORIENTATION_RAI => (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0)
#     ,ORIENTATION_IRP => (0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0)
#     ,ORIENTATION_SRP => (0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
#     ,ORIENTATION_ALI => (0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
#     ,ORIENTATION_PRS => (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
#     ,ORIENTATION_RSP => (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
#     ,ORIENTATION_PRI => (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
#     ,ORIENTATION_IRA => (0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0)
#     ,ORIENTATION_SLA => (0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0)
#     ,ORIENTATION_IAR => (0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0)
#     ,ORIENTATION_PIR => (0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
#     ,ORIENTATION_ILP => (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0)
#     ,ORIENTATION_ALS => (0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
#     ,ORIENTATION_PIL => (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
#     ,ORIENTATION_PSL => (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
#     ,ORIENTATION_RSA => (-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
#     ,ORIENTATION_IPR => (0.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
#     ,ORIENTATION_RIP => (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)
#     ,ORIENTATION_LAI => (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0)
#     ,ORIENTATION_PLI => (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
#     ,ORIENTATION_LSA => (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
#     ,ORIENTATION_SPR => (0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
#     ,ORIENTATION_IAL => (0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0)
#     ,ORIENTATION_SAL => (0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0)
#     ,ORIENTATION_ARS => (0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
#     ,ORIENTATION_LPS => (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
#     ,ORIENTATION_LSP => (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
#     ,ORIENTATION_RIA => (-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0)
#     ,ORIENTATION_ILA => (0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0)
#     ,ORIENTATION_SRA => (0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0)
#     ,ORIENTATION_RPS => (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
#     ,ORIENTATION_SAR => (0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0)
#     ,ORIENTATION_RPI => (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
#     ,ORIENTATION_LIA => (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0)
#     ,ORIENTATION_IPL => (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
#     ,ORIENTATION_SLP => (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
# )

# all_res=brute_force_find_from_sitk(path_nifti)
# dict_curr=Dict(all_res)
# for el in collect(dict_curr)
#     println(",$(el)")
# end


# save("/home/jakubmitura/projects/MedImage.jl/test_data/my_dict.jld", "my_dict", dict_curr)
# # krowa now create a dictionary that will map orientation vector of numbers into orientation enum
end
