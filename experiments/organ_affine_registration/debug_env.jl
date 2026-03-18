println("Julia environment check...")
using MedImages
println("MedImages loaded.")
using CUDA
println("CUDA functional: ", CUDA.functional())
if CUDA.functional()
    println("Device: ", CUDA.name(CUDA.device()))
end
