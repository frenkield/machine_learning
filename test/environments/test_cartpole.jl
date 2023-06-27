using Test
include("../../src/environments/cartpole.jl")

function test_1()

    cartpole = CartPole()

    reset!(cartpole, zeros(4))
    @test cartpole.state == zeros(4)
    
    step!(cartpole, -1.0)
    @test cartpole.state ≈ [0.0, -0.19512194, 0.0, 0.29268292] atol=1e-7
    
    step!(cartpole, -1.0)
    @test cartpole.state ≈ [-0.00390244, -0.3902439 , 0.00585366, 0.58536583] atol=1e-7
    
    step!(cartpole, -1.0)
    @test cartpole.state ≈ [-0.01170732, -0.5854474, 0.01756098, 0.879887] atol=1e-7
    
    step!(cartpole, -1.0)
    @test cartpole.state ≈ [-0.02341626, -0.78080344, 0.03515872, 1.1780386] atol=1e-7
    
end

function test_2()

    cartpole = CartPole()
    reset!(cartpole, [-0.02719117, 0.04814354, 0.03732376, -0.00290003])

    step!(cartpole, 1.0)
    @test cartpole.state ≈ [-0.0262283, 0.24271089, 0.03726576, -0.28357714] atol=1e-7

    step!(cartpole, 1.0)
    @test cartpole.state ≈ [-0.02137408, 0.43728206, 0.03159421, -0.56427765] atol=1e-7

    step!(cartpole, -1.0)
    @test cartpole.state ≈ [-0.01262844, 0.24173138, 0.02030866, -0.2618109] atol=1e-7

    step!(cartpole, -1.0)
    @test cartpole.state ≈ [-0.00779381, 0.04632551, 0.01507244, 0.0372078] atol=1e-7

    step!(cartpole, 1.0)
    @test cartpole.state ≈ [-0.0068673 , 0.2412281 , 0.0158166 , -0.25068176] atol=1e-7
    
end

@testset "Gym CartPole" begin
    test_1()
    test_2()
end
