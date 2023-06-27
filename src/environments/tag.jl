abstract type Environment end

mutable struct Position
    x::Float64
    Position() = new(0.0)
    Position(x::Float64) = new(x)
end

function distance(position_1::Position, position_2::Position)
    norm(position_1.x - position_2.x)
end

Base.@kwdef mutable struct Tag <: Environment

    player = Position(1.0)
    pursuer = Position()

    time = 0.0
    status = :playing

    direction = 1.0

end

state(tag::Tag) = [tag.player.x, tag.pursuer.x, sign(tag.player.x - tag.pursuer.x)]

function step!(tag::Tag, action::Vector{Float64})
    step!(tag, first(action))
end

function step!(tag::Tag, action::Number)

    tag.time += 1.0

    tag.player.x += action * 0.05
    straight_chase!(tag)

    if distance(tag.player, tag.pursuer) < 0.1
        tag.status = :tagged

    elseif norm(tag.player.x) > 3.0
        tag.status = :out_of_bounds

    elseif tag.time > 5000
        tag.status = :timed_out
        # @show "timed_out"
    end

    # @show action, state(tag)

end

function reward(tag::Tag)

    r = 1.0

    if tag.status != :playing
        # @show tag.status
    end

    if tag.status == :tagged
        r = 0.0
    elseif tag.status == :out_of_bounds
        r = 0.0
    elseif tag.status == :timed_out
        r = 1.0
    end

    return r
end

function reset!(tag::Tag)

    tag.time = 0.0
    tag.status = :playing

    tag.player.x = rand([1.0, -1.0]) * (rand() * 0.7 + 0.3)

    tag.pursuer.x = 0.0
    tag.direction = rand([1.0, -1.0])

end

state_is_terminal(tag::Tag) = tag.status != :playing

"""
    action_for_environment(action_index)

Convert action index to tag action: -1 for left or 1 for right.
The action index is specifically the index of the value chosen (at random)
from the actor MLP's outputs.
"""
function action_for_environment(::Tag, action_index::Int)::Vector{Float64}
    return [float(action_index * 2 - 3)]
end

function sin_chase!(tag::Tag)
    tag.pursuer.x = tag.pursuer_direction * sin(tag.time / 100)
end

function straight_chase!(tag::Tag)

    tag.pursuer.x += tag.direction / 50

    if norm(tag.pursuer.x) > 1
        tag.direction *= -1
    end

end
