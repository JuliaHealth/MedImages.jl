using Pkg
Pkg.activate("/home/user/MedImages.jl")
using Statistics, LinearAlgebra

vis_dir = "elsarticle/dosimetry/vis_results"
patients = filter(isdir, readdir(vis_dir, join=true))

println("Final Performance Comparison: Model vs Baseline")
println("-"^60)
println(rpad("Patient", 40), " | ", rpad("Model Corr", 10), " | ", rpad("Base Corr", 10))

for pat in patients
    name = basename(pat)
    dims_txt = read(joinpath(pat, "dims.txt"), String)
    dims = Tuple(parse.(Int, split(dims_txt, ",")))
    
    function load_bin(f)
        d = Array{Float32}(undef, dims)
        read!(joinpath(pat, f), d)
        return reshape(d, :)
    end
    
    pred = load_bin("pred.bin")
    orig = load_bin("orig.bin")
    approx = load_bin("approx.bin")
    
    # Pearson Correlation (Scale invariant)
    corr_model = cor(pred, orig)
    corr_base = cor(approx, orig)
    
    # Normalized MAE (Rescale each to [0,1] to compare structural error)
    function nmae(a, b)
        a_norm = (a .- minimum(a)) ./ (maximum(a) - minimum(a) + 1f-6)
        b_norm = (b .- minimum(b)) ./ (maximum(b) - minimum(b) + 1f-6)
        return mean(abs.(a_norm .- b_norm))
    end
    
    println(rpad(name[1:min(end,38)], 40), " | ", 
            rpad(round(corr_model, digits=4), 10), " | ", 
            rpad(round(corr_base, digits=4), 10))
end
