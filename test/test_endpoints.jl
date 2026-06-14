using LinearAlgebra, Statistics
function extract_endpoints(img::Array{Float32, 3}, threshold=0.1f0)
    coords = findall(img .> threshold)
    if isempty(coords)
        return [16.0, 16.0, 16.0], [16.0, 16.0, 16.0]
    end
    N = length(coords)
    X = zeros(Float64, N, 3)
    for i in 1:N
        X[i, 1] = coords[i][1] - 1
        X[i, 2] = coords[i][2] - 1
        X[i, 3] = coords[i][3] - 1
    end
    
    μ = mean(X, dims=1)
    X_c = X .- μ
    Cov = (X_c' * X_c) ./ (N - 1)
    
    F = eigen(Cov)
    idx = argmax(real.(F.values))
    principal_axis = real.(F.vectors[:, idx])
    
    projections = X_c * principal_axis
    min_idx = argmin(projections[:, 1])
    max_idx = argmax(projections[:, 1])
    
    p1 = X[min_idx, :]
    p2 = X[max_idx, :]
    
    return p1, p2
end

img = zeros(Float32, 32, 32, 32)
for i in 10:20
    img[i, 16, 16] = 1.0f0
end
p1, p2 = extract_endpoints(img)
println(p1, " ", p2)
