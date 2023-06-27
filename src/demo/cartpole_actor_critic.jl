using Test
using Random
using Plots
using Plots.PlotMeasures

include("../../src/environments/cartpole.jl")
include("../../src/environments/cartpole_renderer.jl")
include("../../src/mlp/mlp.jl")
include("../../src/mlp/propagate.jl")
include("../../src/mlp/actions.jl")
include("../../src/train/networks.jl")
include("../../src/train/actor_critic.jl")

function train_cartpole(training_iterations=20000)

    Random.seed!()
    actor_network = ActorNetwork()
    critic_network = CriticNetwork()

    cartpole = CartPole()
    reset!(cartpole)

    train_actor_critic!(actor_network, critic_network, cartpole, iterations=training_iterations)

    return actor_network

end

function render_cartpole(actor_network::ActorNetwork, frames=100)

    cartpole = CartPole()
    reset!(cartpole)

    for i in 1:frames

        scene = plot(xlims=(-3, 3), ylims=(-0.25, 1.25), legend=false, yshowaxis=false,
                     aspect_ratio=:equal, size=(1000,400))

        plot_cartpole(cartpole)
        display(scene)

        action = execute_action!(actor_network.propagator, cartpole.state, deterministic=true)
        step!(cartpole, action_for_environment(cartpole, last(action)))

        if cartpole.out_of_bounds
            println("Resetting cartpole")
            reset!(cartpole)
        end

        sleep(0.01)

    end

end

function animate_cartpole(actor_network::ActorNetwork, frames=300)

    cartpole = CartPole()
    reset!(cartpole)

    animation = @animate for i in 1:frames

        plot(xlims=(-2, 2), ylims=(-0.25, 1.25), legend=false, yshowaxis=false,
             aspect_ratio=:equal, size=(1000,400))

        plot_cartpole(cartpole)

        action = execute_action!(actor_network.propagator, cartpole.state, deterministic=true)
        step!(cartpole, action_for_environment(cartpole, last(action)))

        if cartpole.out_of_bounds
            reset!(cartpole)
        end

    end

    gif(animation, "cartpole_actor_critic.gif", fps = 30)

end
