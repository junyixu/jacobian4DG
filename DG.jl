# From
# https://trixi-framework.github.io/Trixi.jl/dev/tutorials/scalar_linear_advection_1d/
using Trixi
using LinearAlgebra

coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0  # maximum coordinate

initial_condition_sine_wave(x) = 1.0 + 0.5 * sin(π*x)



n_elements = 16 # number of elements
dx = (coordinates_max - coordinates_min) / n_elements # length of one element

polydeg = 3 #= polynomial degree = N =#
basis = LobattoLegendreBasis(polydeg)

nodes = basis.nodes

weights = basis.weights

integral = sum(nodes.^3 .* weights)

x = Matrix{Float64}(undef, length(nodes), n_elements)


for element in 1:n_elements
    x_l = coordinates_min + (element - 1) * dx + dx/2 # middle of each element
    for i in 1:length(nodes)
        ξ = nodes[i] # nodes in [-1, 1]
        x[i, element] = x_l + dx/2 * ξ
    end
end

u0 = initial_condition_sine_wave.(x)

M = diagm(weights)
B = diagm([-1; zeros(polydeg - 1); 1])
D = basis.derivative_matrix
surface_flux = flux_lax_friedrichs

include("rhs.jl")
