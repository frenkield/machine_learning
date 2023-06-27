
abstract type Network end

weights(network::Network) = network.mlp.weights
weights_gradients(network::Network) = network.propagator.weights_gradients

mutable struct ActorNetwork <: Network

    mlp::MLP
    propagator::Propagator
    action_probabilities::Vector{Float64}

    function ActorNetwork(input_count::Int, output_count::Int)
        mlp = MLP([input_count, 64, 64, output_count])
        propagator = Propagator(mlp)
        orthogonalize_weights!(mlp)
        new(mlp, propagator)
    end

    function ActorNetwork()
        return ActorNetwork(4, 2)
    end

end

function action_for_state!(actor_network::ActorNetwork, state::Vector{Float64})

    action_values = forward_propagate!(actor_network.propagator, state)
    action_probabilities, action_index = choose_action(action_values)

    actor_network.action_probabilities = action_probabilities
    return action_index

end

# Here we use the least squares gradients to compute the log probability gradients.
function generate_log_probability_gradients!(actor_network::ActorNetwork, action_index::Int)
    probability_selector = -actor_network.action_probabilities
    probability_selector[action_index] += 1.0
    generate_least_squares_gradients!(actor_network.propagator, probability_selector)
end

# ========================================================

mutable struct CriticNetwork <: Network

    mlp::MLP
    propagator::Propagator

    function CriticNetwork(input_count::Int)
        mlp = MLP([input_count, 64, 64, 1])
        propagator = Propagator(mlp)
        orthogonalize_weights!(mlp)
        new(mlp, propagator)
    end

    function CriticNetwork()
        return CriticNetwork(4)
    end

end

function state_value!(critic_network::CriticNetwork, state)
    return first(forward_propagate!(critic_network.propagator, state))
end

function generate_mlp_gradients!(critic_network::CriticNetwork)
    generate_mlp_gradients!(critic_network.propagator)
end
