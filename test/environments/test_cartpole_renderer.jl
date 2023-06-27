using Plots
using Random

include("../../src/environments/cartpole.jl")
include("../../src/environments/cartpole_actor_mlp.jl")
include("../../src/environments/cartpole_renderer.jl")

function simple_test()

    cartpole = CartPole()
    cartpole.state[3] = 0.1

    for i in 1:200

        scene = plot(xlims=(-6, 6), ylims=(-6, 6), legend=false, aspect_ratio=:equal)

        plot_cartpole(cartpole)
        step!(cartpole)

        display(scene)
        sleep(0.05)

    end
end

function policy_mlp_test_1()

    cartpole = CartPole()
    policy_mlp = cartpole_policy_mlp()
    cartpole.state = [-0.02908801, -0.03668493, 0.0233334,  -0.01141027]

    for i in 1:100

        scene = plot(xlims=(-1, 1), ylims=(-1, 1), legend=false, aspect_ratio=:equal)
        plot_cartpole(cartpole)
        display(scene)

        action = cartpole_action!(policy_mlp, cartpole.state)

        @show cartpole.state, action
        step!(cartpole, action)

        sleep(0.1)

    end
end

function policy_mlp_test_2()

    Random.seed!()
    cartpole = CartPole()
    reset!(cartpole)
    policy_mlp = cartpole_policy_mlp()

    for i in 1:5000

        scene = plot(xlims=(-3, 3), ylims=(-0.25, 1.25), legend=false, aspect_ratio=:equal)
        plot_cartpole(cartpole)
        display(scene)

        action = cartpole_action!(policy_mlp, cartpole.state)
        step!(cartpole, last(action))

        if cartpole.out_of_bounds
            reset!(cartpole)
        end

        sleep(0.001)

    end
end

policy_mlp_test_2()
