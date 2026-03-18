module RegistrationModel

using Lux
using Random
using NNlib

export MultiScaleCNN

"""
    MultiScaleCNN(in_channels::Int, num_organs::Int)

A 3D CNN architecture for predicting organ-specific affine transformation parameters.

# Architecture
- **Multi-Scale Convolutions**: Parallel branches with kernel sizes (3,3,3), (5,5,5), and (7,7,7) to capture features at different spatial resolutions.
- **Global Pooling**: Reduces spatial dimensions after feature extraction.
- **MLP Head**: Projects features to `num_organs * 15` parameters.

# Output Parameters (per organ)
1.  **Rotation (3)**: Tanh activation scaled by π (radians).
2.  **Translation (3)**: Linear output.
3.  **Scale (3)**: Exponential activation (positive).
4.  **Shear (3)**: Tanh activation.
5.  **Center (3)**: Linear output.

# Arguments
- `in_channels`: Number of input channels (e.g., 2 for CT + Atlas).
- `num_organs`: Number of organs to predict parameters for.
"""
struct MultiScaleCNN{C, D} <: Lux.AbstractLuxLayer
    conv_branches::C
    dense_layers::D
    num_organs::Int
end

function MultiScaleCNN(in_channels::Int, num_organs::Int)
    # 3D Convolution branches
    # Branch 1: 3x3x3
    b1 = Chain(
        Conv((3, 3, 3), in_channels => 32, relu; stride=2, pad=1),
        Conv((3, 3, 3), 32 => 64, relu; stride=2, pad=1)
    )

    # Branch 2: 5x5x5
    b2 = Chain(
        Conv((5, 5, 5), in_channels => 32, relu; stride=2, pad=2),
        Conv((5, 5, 5), 32 => 64, relu; stride=2, pad=2)
    )

    # Branch 3: 7x7x7
    b3 = Chain(
        Conv((7, 7, 7), in_channels => 32, relu; stride=2, pad=3),
        Conv((7, 7, 7), 32 => 64, relu; stride=2, pad=3)
    )

    # After stride 2 twice, size is W/4, H/4, D/4.
    # We GlobalAvgPool each branch.

    branches = Parallel(vcat,
        Chain(b1, GlobalMeanPool(), FlattenLayer()),
        Chain(b2, GlobalMeanPool(), FlattenLayer()),
        Chain(b3, GlobalMeanPool(), FlattenLayer())
    )
    # Output of branches: 64 + 64 + 64 = 192 features (per batch item)

    # MLP Head
    # Output size: num_organs * 15 parameters
    # Params: Rotation(3), Translation(3), Scale(3), Shear(3), Center(3)
    out_dim = num_organs * 15

    dense = Chain(
        Dense(192 => 512, relu),
        Dense(512 => 512, relu),
        Dense(512 => 256, relu),
        Dense(256 => out_dim)
    )

    return MultiScaleCNN(branches, dense, num_organs)
end

function Lux.initialparameters(rng::AbstractRNG, layer::MultiScaleCNN)
    return (
        conv_branches = Lux.initialparameters(rng, layer.conv_branches),
        dense_layers = Lux.initialparameters(rng, layer.dense_layers)
    )
end

function Lux.initialstates(rng::AbstractRNG, layer::MultiScaleCNN)
    return (
        conv_branches = Lux.initialstates(rng, layer.conv_branches),
        dense_layers = Lux.initialstates(rng, layer.dense_layers)
    )
end

function (m::MultiScaleCNN)(x, ps, st)
    # x is (W, H, D, C, Batch)

    # Extract features
    feats, st_conv = m.conv_branches(x, ps.conv_branches, st.conv_branches)

    # Predict raw parameters
    raw_out, st_dense = m.dense_layers(feats, ps.dense_layers, st.dense_layers)

    # Apply constraints / Reshape
    # Output: (Num_Organs * 15, Batch)
    # We reshape to (15, Num_Organs, Batch) for easier processing

    batch_size = size(raw_out, 2)
    reshaped = reshape(raw_out, 15, m.num_organs, batch_size)

    # Apply Activations to constrain parameters
    # Indices:
    # 1-3: Rotation (Radians, e.g. tanh * pi)
    # 4-6: Translation (Unbounded or scaled? Let's leave linear for now or Softsign)
    # 7-9: Scale (Must be positive, e.g. exp)
    # 10-12: Shear (tanh * const)
    # 13-15: Center (Unbounded or relative?)

    # NOTE: Zygote needs pure array operations.

    # Slices (using view or getindex)
    rot = tanh.(reshaped[1:3, :, :]) .* Float32(π)
    trans = reshaped[4:6, :, :] # Linear
    scale_p = exp.(reshaped[7:9, :, :]) # Positive
    shear = tanh.(reshaped[10:12, :, :]) # -1 to 1
    center = reshaped[13:15, :, :] # Linear

    # Reassemble
    # To keep AD happy, we can cat
    final_params = cat(rot, trans, scale_p, shear, center; dims=1)

    return final_params, (conv_branches=st_conv, dense_layers=st_dense)
end

end # module
