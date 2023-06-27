using LinearAlgebra

neighbors(A) = zip(A[1:end-1], A[2:end])
softmax(x) = exp.(x) / sum(exp.(x))

struct Layer

    values_with_bias::Vector{Float64}
    values::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}

    function Layer(layer_length)
        values_with_bias = [zeros(layer_length)..., 1]
        values = @view values_with_bias[1:end-1]
        new(values_with_bias, values)
    end
    
end

Base.length(layer::Layer) = length(layer.values)
length_with_bias(layer::Layer) = length(layer.values_with_bias)

struct Weights

    values_with_bias::Matrix{Float64}
    values::SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}

    function Weights(input_layer::Layer, output_layer::Layer)
        values_with_bias = randn(length(output_layer), length_with_bias(input_layer))
        values = @view values_with_bias[:, 1:end-1]
        values_with_bias[:, end] .= 0
        new(values_with_bias, values)
    end

end

bias_values(weights::Weights) = weights.values_with_bias[:, end]

function orthogonalize!(weights::Weights)
    s = svd(weights.values)
    if axes(weights.values, 1) > axes(weights.values, 2)
        weights.values .= s.U
    else
        weights.values .= s.Vt
    end
end

function set_values!(weights::Weights, values::Matrix{Float64}, bias::Vector{Float64})
    weights.values .= values
    weights.values_with_bias[:, end] .= bias
end

struct MLP

    layers::Vector{Layer}
    weights::Vector{Weights}
    activation::Function
    deactivation::Function

    function MLP(layer_lenghts::Vector{Int};
                 activation=tanh, deactivation=tanh_prime_back_propagation)

        layers = map(l -> Layer(l), layer_lenghts)
        weights = map(in_out -> Weights(in_out...), neighbors(layers))
        new(layers, weights, activation, deactivation)
        
    end

end

output(mlp::MLP) = mlp.layers[end].values
logits(mlp::MLP) = softmax(output(mlp))

layer_lengths(mlp::MLP) = map(l -> length(l), mlp.layers)

function orthogonalize_weights!(mlp::MLP)
    foreach(w -> orthogonalize!(w), mlp.weights)
end
