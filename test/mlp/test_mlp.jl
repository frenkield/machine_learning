using Test
using Random
using Profile
include("../../src/mlp/mlp.jl")
include("../../src/mlp/propagate.jl")

function test_1()

    Random.seed!(304958)
    mlp = MLP([2, 3, 1])
    propagator = Propagator(mlp, activation=sigmoid)

    samples = [
        ([4.0, 5], [3.0]),
        ([2.0, 1], [10.0]),
    ]

    least_squares_train!(propagator, samples)

    forward_propagate!(propagator, first(samples[1]))
    @test output(mlp) ≈ last(samples[1]) atol=1e-5

    forward_propagate!(propagator, first(samples[2]))
    @test output(mlp) ≈ last(samples[2]) atol=1e-5

end

function test_2()

    mlp = MLP([2, 3, 1])

    Random.seed!(80958)
    values = rand(3, 2)
    bias = rand(3)
    set_values!(mlp.weights[1], values, bias)

    @test mlp.weights[1].values == values
    @test bias_values(mlp.weights[1]) == bias

end

function test_3()

    Random.seed!(87654)
    mlp = MLP([2, 3, 1])
    propagator = Propagator(mlp)

    samples = [
        ([4.0, 5], [3.0]),
        ([2.0, 1], [10.0]),
    ]

    least_squares_train!(propagator, samples)

    forward_propagate!(propagator, first(samples[1]))
    @test output(mlp) ≈ last(samples[1]) atol=1e-5

    forward_propagate!(propagator, first(samples[2]))
    @test output(mlp) ≈ last(samples[2]) atol=1e-5
    
end

function test_4()

    Random.seed!(34674)
    mlp = MLP([2, 10, 2])
    propagator = Propagator(mlp)

    samples = [
        ([10.0, 20], [0.1, 0.2]),
        ([30.0, 40], [-0.3, 0.4])
    ]

    least_squares_train!(propagator, samples)

    forward_propagate!(propagator, first(samples[1]))
    @test output(mlp) ≈ last(samples[1]) atol=1e-5

    forward_propagate!(propagator, first(samples[2]))
    @test output(mlp) ≈ last(samples[2]) atol=1e-5

end

function test_5()

    mlp = MLP([2, 3, 1])
    propagator = Propagator(mlp)

    mlp.weights[1].values_with_bias .= [-1 2 3; 1 2 3; 1 2 3]
    mlp.weights[2].values_with_bias .= [1 -2 3 4]

    forward_propagate!(propagator, [1.0, 1.0])
    @test output(mlp) ≈ [5.999317011389863]

end

function test_orthogonalize_weights_1()

    mlp = MLP([3, 5, 2])

    orthogonalize!(mlp.weights[1])
    @test mlp.weights[1].values' * mlp.weights[1].values ≈ I(3)

    orthogonalize!(mlp.weights[2])
    @test mlp.weights[2].values * mlp.weights[2].values' ≈ I(2)

    mlp = MLP([3, 5, 2])
    orthogonalize_weights!(mlp)
    @test mlp.weights[1].values' * mlp.weights[1].values ≈ I(3)
    @test mlp.weights[2].values * mlp.weights[2].values' ≈ I(2)

end

function test_compute_weight_gradients_1()

    Random.seed!(304958)
    mlp = MLP([3, 4, 2])
    propagator = Propagator(mlp)

    samples = [
        ([1.0, 2, 3], [10.0, 20])
    ]

    generate_least_squares_gradients!(propagator, ones(2))
    @test size(propagator.weights_gradients[1]) == (4, 4)
    @test size(propagator.weights_gradients[2]) == (2, 5)
    @test any(propagator.weights_gradients[1] .!= 0.0)
    @test any(propagator.weights_gradients[2] .!= 0.0)

    generate_least_squares_gradients!(propagator, zeros(2))
    @test all(propagator.weights_gradients[1] .== 0.0)
    @test all(propagator.weights_gradients[2] .== 0.0)

    propagator_1 = Propagator(mlp)
    generate_least_squares_gradients!(propagator_1, [1.0, 0])

    propagator_2 = Propagator(mlp)
    generate_least_squares_gradients!(propagator_2, [0.0, 1])

    propagator_3 = Propagator(mlp)
    generate_least_squares_gradients!(propagator_3, ones(2))

    @test propagator_3.weights_gradients[1] == propagator_1.weights_gradients[1] +
                                               propagator_2.weights_gradients[1]

end

# This test demonstrates that for networks with a single output
# we can can compute all the gradients (for all weights) using
# the least squares gradients.
function test_compute_weight_gradients_2()

    Random.seed!()
    mlp = MLP([2, 13, 3, 1])

    propagator = Propagator(mlp)
    forward_propagate!(propagator, randn(2))
    generate_least_squares_gradients!(propagator, ones(1))

    scale = [rand()]
    propagator_with_scale = Propagator(mlp)
    generate_least_squares_gradients!(propagator_with_scale, scale)

    @test propagator_with_scale.weights_gradients[1] ≈
        scale .* propagator.weights_gradients[1] atol=1e-10

    @test propagator_with_scale.weights_gradients[2] ≈
        scale .* propagator.weights_gradients[2] atol=1e-10

    @test propagator_with_scale.weights_gradients[3] ≈
        scale .* propagator.weights_gradients[3] atol=1e-10

end

function profile_1()
    Random.seed!(304958)
    mlp = MLP([2, 3, 1])
    samples = [
        ([4.0, 5], [3.0]),
        ([2.0, 1], [10.0]),
    ]
    train!(mlp, samples)
    @profile for i in 1:100 train!(mlp, samples) end
end

@testset "MLP and propagation" begin
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_orthogonalize_weights_1()
    test_compute_weight_gradients_1()
    test_compute_weight_gradients_2()
end
