using Pkg
Pkg.activate("/home/user/MedImages.jl")
using ComponentArrays, Serialization, Lux, CUDA, LuxCUDA

function inspect_model(path)
    if !isfile(path); return; end
    println("\nInspecting: ", path)
    θ = deserialize(path)
    println("Keys: ", keys(θ))
    for k in keys(θ)
        v = getproperty(θ, k)
        try
            println("  $k keys: ", keys(v))
            for k2 in keys(v)
                v2 = getproperty(v, k2)
                try
                    println("    $k2 keys: ", keys(v2))
                catch; end
            end
        catch; end
    end
end

inspect_model("elsarticle/dosimetry/model_heavy.jls")
inspect_model("elsarticle/dosimetry/model_ude_no_approx.jls")
inspect_model("elsarticle/dosimetry/model_pure_cnn.jls")
