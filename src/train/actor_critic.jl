# using Statistics

const ALPHA = 0.02
const LAMBDA = 0.5
# const ALPHA = 0.01
# const LAMBDA = 0.1
const ETA = 0.1

# Actor–Critic with Eligibility Traces (continuing)
# Page 277 of http://incompleteideas.net/book/bookdraft2017nov5.pdf
function train_actor_critic!(actor_network::ActorNetwork, critic_network::CriticNetwork,
                             environment; iterations=1000)

    reward_to_go = 0.0
    critic_value = state_value!(critic_network, state(environment))
    iterations_since_reset = 0
    max_steps = 0

    for i in 1:iterations

        iterations_since_reset += 1

        # Do this first since we forward propagate in next_step!().
        generate_mlp_gradients!(critic_network)

        step = next_step!(actor_network, critic_network, environment, critic_value)

        if step.action_is_terminal
            delta = step.reward - reward_to_go - step.critic_value
        else
            delta = step.reward - reward_to_go + step.next_critic_value - step.critic_value
        end

        reward_to_go += ETA * delta

        critic_weight_updates = compute_weight_updates(critic_network)

        generate_log_probability_gradients!(actor_network, step.action_index)
        actor_weight_updates = compute_weight_updates(actor_network)

        update_weights!(critic_network.mlp, critic_weight_updates, delta)
        update_weights!(actor_network.mlp, actor_weight_updates, delta)

        if step.action_is_terminal

            if iterations_since_reset > max_steps
                max_steps = iterations_since_reset
                @info "Max steps acheived: $(max_steps)"
            end

            iterations_since_reset = 0
            reset!(environment)

        end

        critic_value = step.next_critic_value

    end

end

function compute_weight_updates(network::Network)
    return map(
        (w, g) -> LAMBDA * w.values_with_bias .+ g,
        weights(network), weights_gradients(network)
    )
end

function update_weights!(mlp::MLP, weight_udpates::Vector{Matrix{Float64}}, delta::Float64)
    foreach(
        (w, u) -> w.values_with_bias .+= ALPHA * delta * u,
        mlp.weights, weight_udpates
    )        
end

function next_step!(actor_network::ActorNetwork, critic_network::CriticNetwork,
                    environment::Environment, critic_value::Float64)

    step = Step(state(environment))
    step.critic_value = critic_value
    
    step.action_index = action_for_state!(actor_network, state(environment))
    step.action = action_for_environment(environment, step.action_index)
    
    step!(environment, step.action)
    step.reward = reward(environment)
    step.action_is_terminal = state_is_terminal(environment)

    step.next_critic_value = state_value!(critic_network, state(environment))

    return step

end

Base.@kwdef mutable struct Step

    state::Vector{Float64}
    critic_value::Float64

    action::Vector{Float64}
    action_is_terminal = false
    action_index = 0
    reward = 0.0

    next_critic_value::Float64

    Step(environment_state::Vector{Float64}) = new(environment_state) 

end
