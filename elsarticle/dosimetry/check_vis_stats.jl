using Pkg
Pkg.activate("/home/user/MedImages.jl")
using Statistics

vis_dir = "elsarticle/dosimetry/vis_results"
patients = filter(isdir, readdir(vis_dir, join=true))

for pat in patients
    println("Stats for patient: ", basename(pat))
    
    dims_txt = read(joinpath(pat, "dims.txt"), String)
    dims = Tuple(parse.(Int, split(dims_txt, ",")))
    
    function get_stats(file)
        if !isfile(file); return "Missing"; end
        data = Array{Float32}(undef, dims)
        read!(file, data)
        return (min=minimum(data), max=maximum(data), mean=mean(data), p99=quantile(reshape(data, :), 0.99))
    end
    
    println("  Pred:   ", get_stats(joinpath(pat, "pred.bin")))
    println("  Orig:   ", get_stats(joinpath(pat, "orig.bin")))
    println("  Approx: ", get_stats(joinpath(pat, "approx.bin")))
end
