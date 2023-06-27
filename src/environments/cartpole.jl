# Copied/adapted from Python Gym package
# https://github.com/openai/environments/blob/master/environments/envs/classic_control/cartpole.py

abstract type Environment end

Base.@kwdef mutable struct CartPole <: Environment

    state = zeros(4)
    out_of_bounds = false

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    pole_half_length = 0.5
    polemass_length = masspole * pole_half_length
    force_mag = 10.0
    time_step = 0.02

    theta_threshold_radians = 12 * 2 * pi / 360
    x_threshold = 2.4

end

function step!(cartpole::CartPole, action::Vector{Float64})
    step!(cartpole, first(action))
end

function step!(cartpole::CartPole, action::Number)

    x, x_dot, theta, theta_dot = cartpole.state
    force = action == 1.0 ? cartpole.force_mag : -cartpole.force_mag

    costheta = cos(theta)
    sintheta = sin(theta)

    temp = (
        force + cartpole.polemass_length * theta_dot^2 * sintheta
    ) / cartpole.total_mass

    thetaacc = (cartpole.gravity * sintheta - costheta * temp) / (
        cartpole.pole_half_length * (4.0 / 3.0 - cartpole.masspole * costheta^2 / cartpole.total_mass)
    )
    
    xacc = temp - cartpole.polemass_length * thetaacc * costheta / cartpole.total_mass

    x = x + cartpole.time_step * x_dot
    x_dot = x_dot + cartpole.time_step * xacc
    theta = theta + cartpole.time_step * theta_dot
    theta_dot = theta_dot + cartpole.time_step * thetaacc

    cartpole.state = [x, x_dot, theta, theta_dot]

    cartpole.out_of_bounds =
        x < -cartpole.x_threshold || x > cartpole.x_threshold ||
        theta < -cartpole.theta_threshold_radians ||
        theta > cartpole.theta_threshold_radians

end

function reward(cartpole::CartPole)
    return cartpole.out_of_bounds ? 0.0 : 1.0
end

function reset!(cartpole::CartPole, state::Vector{Float64}=randu_05(4))
    cartpole.state = state
    cartpole.out_of_bounds = false
end

state(cartpole::CartPole) = cartpole.state
state_is_terminal(cartpole::CartPole) = cartpole.out_of_bounds
position(cartpole::CartPole) = cartpole.state[1]
pole_angle(cartpole::CartPole) = cartpole.state[3]
randu_05(dims) = rand(dims) .* 0.1 .- 0.05

"""
    action_for_environment(action_index)

Convert action index to cartpole action: 0 for push left, 1 for push right.
The action index is specifically the index of the value chosen (at random)
from the actor MLP's outputs.
"""
function action_for_environment(::CartPole, action_index::Int)::Vector{Float64}
    return [float(action_index - 1)]
end
