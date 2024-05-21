
include("./MedImage_data_struct.jl")
include("./Utils.jl")
include("./Load_and_save.jl")

using Interpolations
using Combinatorics
using JLD

orientation_enum_to_string = Dict(
ORIENTATION_RIP=>"RIP",
ORIENTATION_LIP=>"LIP",
ORIENTATION_RSP=>"RSP",
ORIENTATION_LSP=>"LSP",
ORIENTATION_RIA=>"RIA",
ORIENTATION_LIA=>"LIA",
ORIENTATION_RSA=>"RSA",
ORIENTATION_LSA=>"LSA",
ORIENTATION_IRP=>"IRP",
ORIENTATION_ILP=>"ILP",
ORIENTATION_SRP=>"SRP",
ORIENTATION_SLP=>"SLP",
ORIENTATION_IRA=>"IRA",
ORIENTATION_ILA=>"ILA",
ORIENTATION_SRA=>"SRA",
ORIENTATION_SLA=>"SLA",
ORIENTATION_RPI=>"RPI",
ORIENTATION_LPI=>"LPI",
ORIENTATION_RAI=>"RAI",
ORIENTATION_LAI=>"LAI",
ORIENTATION_RPS=>"RPS",
ORIENTATION_LPS=>"LPS",
ORIENTATION_RAS=>"RAS",
ORIENTATION_LAS=>"LAS",
ORIENTATION_PRI=>"PRI",
ORIENTATION_PLI=>"PLI",
ORIENTATION_ARI=>"ARI",
ORIENTATION_ALI=>"ALI",
ORIENTATION_PRS=>"PRS",
ORIENTATION_PLS=>"PLS",
ORIENTATION_ARS=>"ARS",
ORIENTATION_ALS=>"ALS",
ORIENTATION_IPR=>"IPR",
ORIENTATION_SPR=>"SPR",
ORIENTATION_IAR=>"IAR",
ORIENTATION_SAR=>"SAR",
ORIENTATION_IPL=>"IPL",
ORIENTATION_SPL=>"SPL",
ORIENTATION_IAL=>"IAL",
ORIENTATION_SAL=>"SAL",
ORIENTATION_PIR=>"PIR",
ORIENTATION_PSR=>"PSR",
ORIENTATION_AIR=>"AIR", 
ORIENTATION_ASR=>"ASR",
ORIENTATION_PIL=>"PIL",
ORIENTATION_PSL=>"PSL",
ORIENTATION_AIL=>"AIL", 
ORIENTATION_ASL=>"ASL")

string_to_orientation_enum = Dict(value => key for (key, value) in orientation_enum_to_string)
sitk = pyimport_conda("SimpleITK","simpleITK")


function change_image_orientation(path_nifti, orientation)
    # Read the image
    image = sitk.ReadImage(path_nifti)
    
    # print("oroginal orientation $(image.GetDirection()) origin $(image.GetOrigin()) size $(image.GetSize())  \n")
    # Create a DICOMOrientImageFilter
    orient_filter = sitk.DICOMOrientImageFilter()

    # Set the desired orientation
    orient_filter.SetDesiredCoordinateOrientation(orientation)

    # Apply the filter to the image
    oriented_image = orient_filter.Execute(image)
    return oriented_image
    # Write the oriented image back to the file
    # sitk.WriteImage(oriented_image, path_nifti)

end

"""
force solution get all direction combinetions put it in sitk and try all possible ways to permute and reverse axis to get the same result as sitk then save the result in json or sth and use; do the same with the origin
Additionally in similar manner save all directions in a form of a vector and associate it with 3 letter codes
"""





function brute_force_find_perm_rev(sitk_image1_arr,sitk_image2_arr)

    comb =collect(combinations([1,2,3]))
    push!(comb,[])

    perm=collect(permutations([1,2,3]))
    perm = filter(x -> x != [1,2,3], perm)
    push!(perm,[])
    for p in perm
        for c in comb
            if(length(p)>0)
                curr=permutedims(sitk_image2_arr,(p[1],p[2],p[3]))
            else
                curr=sitk_image2_arr    
            end    

            if (length(c)==1)
                curr=reverse(curr;dims=c[1])  
            elseif (length(c)>1)
                curr=reverse(curr;dims=Tuple(c))    
            end
            if(curr==sitk_image1_arr)
                # println("Found")
                # println(p)
                # println(c)
                return (p,c)
            end
            
        end
    end    

end    

function establish_orginn_transformation(sitk_image1,sitk_image2,sitk_image1_arr)
    sizz=size(sitk_image1_arr)
    origin1=sitk_image1.GetOrigin()
    origin2=sitk_image2.GetOrigin()
    spacing1=sitk_image1.GetSpacing()
    spacing2=sitk_image2.GetSpacing()

    sizzz=[(spacing1[1]*(sizz[3]-1)),(spacing1[2]*(sizz[2]-1)),(spacing1[3]*(sizz[1]-1))   ]

    res=[[],[],[]]
    for op in ['+','-',' ']
        for origin_axis in [1,2,3]
            for sizz_axis in range(1,3)
                for flip in [1,-1]
                    loc=0.099123887
                    # p=copy(origin1)
                    if(op=='+')
                        loc=origin1[origin_axis]+sizzz[sizz_axis]
                    elseif(op=='-')
                        loc=origin1[origin_axis]-sizzz[sizz_axis]
                    else
                        loc=origin1[origin_axis]
                    end
                    if((loc*flip)==origin2[origin_axis])
                        res[origin_axis]=[op,origin_axis,sizz_axis,flip]
                    end
                end

            end 
        end   
    end
    print("\n rrrrres $(res) \n")
    # if(isapprox(p,collect(origin2); atol=0.1) )
    #     println("Found origin transformation $(i)")

    # # print("jjjj $(opts) origin2 $(origin2) ")
    diff=collect(origin1)-collect(origin2)
    # print("\n ooooo origin1 $(origin1) origin2 $(origin2) spacing1 $(spacing1) spacing2 $(spacing2) diff $(diff)  sizzz $(sizzz) \n")

    return res
end



function brute_force_find_from_sitk_single(path_nifti,or_enum_1::Orientation_code, or_enum_2::Orientation_code)
    or_enum_1_str=orientation_enum_to_string[or_enum_1]
    or_enum_2_str=orientation_enum_to_string[or_enum_2]
    sitk_image1 = change_image_orientation(path_nifti, or_enum_1_str)
    sitk_image2 = change_image_orientation(path_nifti, or_enum_2_str)

    sitk_image1_arr=sitk.GetArrayFromImage(sitk_image1)
    sitk_image2_arr=sitk.GetArrayFromImage(sitk_image2)

    sitk_image1_arr=permutedims(sitk_image1_arr,(3,2,1))
    sitk_image1_arr=permutedims(sitk_image1_arr,(3,2,1))

    #find what permutation of axis and reversing is needed to get the same result as in sitk
    p,c=brute_force_find_perm_rev(sitk_image1_arr,sitk_image2_arr)
    # get idea how to get ransformed origin
    origin_perm=establish_orginn_transformation(sitk_image1,sitk_image2,sitk_image1_arr)
    return ((or_enum_1,or_enum_2),(p,c,origin_perm))

end



function brute_force_find_from_sitk(path_nifti)
    opts=[]
    for value_out in collect(instances(Orientation_code))
        for value_in in collect(instances(Orientation_code))
            opts=push!(opts,(value_out,value_in))
        end    
    end

    return map(el-> brute_force_find_from_sitk_single(path_nifti,el[1] ,el[2]),opts)
end

path_nifti = "/home/jakubmitura/projects/MedImage.jl/test_data/volume-0.nii.gz"

all_res=brute_force_find_from_sitk(path_nifti)
dict_curr=Dict(all_res)
save("/home/jakubmitura/projects/MedImage.jl/test_data/my_dict.jld", "my_dict", dict_curr)
# krowa now create a dictionary that will map orientation vector of numbers into orientation enum