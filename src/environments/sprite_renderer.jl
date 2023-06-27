using Plots

function plot_rectangle(x::Float64, y::Float64;
                        width::Float64=0.3, height::Float64=0.2, color=:red)

    position = (x, y)
    half_width = width / 2
    half_height = height / 2

    rectangle = Shape([
        (position .+ (-half_width, -half_height)),
        (position .+ (half_width, -half_height)),
        (position .+ (half_width, half_height)),
        (position .+ (-half_width, half_height))
    ])

    plot!(rectangle, fillcolor=color)

end

function plot_stick(position::Tuple{Float64, Float64}, angle::Float64; length::Float64=1.0)
    end_position = position .+ (sin(angle) * length, cos(angle) * length)
    pole = Shape([position, end_position])
    plot!(pole, fillcolor=:green, linewidth=5.0)
end
