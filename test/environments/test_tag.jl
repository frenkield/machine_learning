using Random

include("../../src/environments/tag.jl")
include("../../src/environments/sprite_renderer.jl")
include("../../src/mlp/mlp.jl")
include("../../src/mlp/propagate.jl")
include("../../src/mlp/actions.jl")
include("../../src/train/networks.jl")
include("../../src/train/actor_critic.jl")

function train_tag_1(training_iterations=1000)

    Random.seed!()

    state_length = 3
    action_length = 2

    actor_network = ActorNetwork(state_length, action_length)
    critic_network = CriticNetwork(state_length)

    tag = Tag()
    reset!(tag)

    train_actor_critic!(actor_network, critic_network, tag, iterations=training_iterations)
    @info "Done training"

    return actor_network

end

function render_tag_1(actor_network::ActorNetwork; frames::Int=100)

    tag = Tag()
    reset!(tag)

    for i in 1:frames

        scene = plot(xlims=(-3, 3), ylims=(-0.25, 1.25), legend=false, aspect_ratio=:equal)
        plot_rectangle(tag.player.x, 0.0, color=:blue)
        plot_rectangle(tag.pursuer.x, 0.0, color=:red)
        display(scene)

        action = execute_action!(actor_network.propagator, state(tag), deterministic=true)
        
        step!(tag, action_for_environment(tag, last(action)))

        if state_is_terminal(tag)
            # println("Game over: $(tag.status)")
            reset!(tag)
        end
    
        sleep(0.02)

    end

end

train_tag_1()
