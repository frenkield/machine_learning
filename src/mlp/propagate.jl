sigmoid(x) = 1 / (1 + exp(-x))
sigmoid_prime_back_propagation(layer::Layer) = layer.values .* (1 .- layer.values)
tanh_prime_back_propagation(layer::Layer) = 1 .- layer.values.^2

struct Propagator

    mlp::MLP
    activation::Function
    deactivation::Function
    differences::Vector{Vector{Float64}}
    weights_gradients::Vector{Matrix{Float64}}

    function Propagator(mlp::MLP; activation=tanh, deactivation=tanh_prime_back_propagation)
        differences = map(l -> similar(l.values), mlp.layers[2:end])
        weights_gradients = map(w -> similar(w.values_with_bias), mlp.weights)
        new(mlp, activation, deactivation, differences, weights_gradients)
    end
end

function forward_propagate!(propagator::Propagator, inputs::Vector{Float64})

    mlp = propagator.mlp
    layers = mlp.layers
    weights = mlp.weights

    layers[1].values .= inputs

    for i in 2:lastindex(layers)-1
        layers[i].values .= weights[i-1].values_with_bias * layers[i-1].values_with_bias
        layers[i].values .= propagator.activation.(layers[i].values)
    end

    layers[end].values .= weights[end].values_with_bias * layers[end-1].values_with_bias
    return output(mlp)

end

function least_squares_back_propagate!(propagator::Propagator,
                                       expected_outputs::Vector{Float64};
                                       learning_rate=0.1)

    mlp = propagator.mlp
    
    output_difference = mlp.layers[end].values - expected_outputs
    generate_least_squares_gradients!(propagator, output_difference)

    for i in eachindex(propagator.weights_gradients)
        mlp.weights[i].values_with_bias .+= -learning_rate * propagator.weights_gradients[i]
    end

end

function generate_least_squares_gradients!(propagator::Propagator,
                                           output_difference::Vector{Float64})

    mlp = propagator.mlp
    layers = mlp.layers
    weights = mlp.weights
    differences = propagator.differences
    deactivation = propagator.deactivation
    weights_gradients = propagator.weights_gradients

    differences[end] .= output_difference

    for i in lastindex(differences):-1:2
        differences[i-1] .= deactivation(layers[i]) .* (weights[i].values' * differences[i])
    end

    for i in eachindex(differences)
        weights_gradients[i] .= differences[i] * layers[i].values_with_bias'
    end

end

function generate_mlp_gradients!(propagator::Propagator)
    length(propagator.mlp.layers[end]) == 1 ||
        @error "MLP gradients only work for single-output layers."
    generate_least_squares_gradients!(propagator, ones(1))
end

function least_squares_train!(propagator::Propagator,
                              samples::Vector{Tuple{Vector{Float64}, Vector{Float64}}};
                              iterations=1000)

    for i in 1:iterations
        for sample in samples
            forward_propagate!(propagator, first(sample))
            least_squares_back_propagate!(propagator, last(sample))
        end
    end

end
