using Plots

function plot_box(position::Tuple{Float64, Float64};
                  width::Float64=0.3, height::Float64=0.2)

    half_width = width / 2
    half_height = height / 2

    box = Shape([
        (position .+ (-half_width, -half_height)),
        (position .+ (half_width, -half_height)),
        (position .+ (half_width, half_height)),
        (position .+ (-half_width, half_height))
    ])

    plot!(box, fillcolor=:red)

end

function plot_pole(position::Tuple{Float64, Float64}, angle::Float64; length::Float64=1.0)
    end_position = position .+ (sin(angle) * length, cos(angle) * length)
    pole = Shape([position, end_position])
    plot!(pole, fillcolor=:green, linewidth=5.0)
end

function plot_cartpole(cartpole::CartPole)
    cartpole_position = (position(cartpole), 0.0)
    plot_box(cartpole_position)
    plot_pole(cartpole_position, pole_angle(cartpole))
end
