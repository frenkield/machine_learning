function execute_action!(propagator::Propagator, environment_state::Vector{Float64};
                         deterministic=false)
    forward_propagate!(propagator, environment_state)
    return choose_action(output(propagator.mlp); deterministic)
end

function choose_action(mlp_output::AbstractArray{Float64}; deterministic=false)

    if any(isnan.(mlp_output))
        throw("NaNs in network output.")
    end

    probabilities = action_probabilities(mlp_output)

    if deterministic
        action_index = argmax(probabilities)
    else
        action_index = categorical_probability(probabilities)
    end

    return probabilities, action_index

end

function action_probabilities(mlp_output::AbstractArray{Float64})
    return softmax(mlp_output)
end

function categorical_probability(probabilities::Vector{Float64})
    sample = rand()
    total = probabilities[1]
    selection = 1
    while sample >= total
        selection += 1
        total += probabilities[selection]
    end
    return selection
end
