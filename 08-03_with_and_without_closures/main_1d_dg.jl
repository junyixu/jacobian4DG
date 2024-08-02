using Trixi
using Enzyme

# equation with a advection_velocity of `1`.
advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# create DG solver with flux lax friedrichs and LGL basis
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

# distretize domain with `TreeMesh`
coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4, # number of elements = 2^4
                n_cells_max=30_000)

# create initial condition and semidiscretization
initial_condition_sine_wave(x, t, equations) = SVector(1.0 + 0.5 * sin(pi * sum(x - equations.advection_velocity * t)))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine_wave, solver)

# %%
include("my_rhs.jl")
include("ad_functions.jl")
# %%

@time J1 = jacobian_enzyme_forward_closure(semi); #  0.000888 seconds (3.76 k allocations: 278.953 KiB)
@time J2 = jacobian_enzyme_reverse_closure(semi); #  0.003550 seconds (10.48 k allocations: 609.953 KiB)
@time J3 = jacobian_enzyme_forward(semi); # 0.000191 seconds (1.42 k allocations: 148.625 KiB)
@time J4 = jacobian_enzyme_reverse(semi); # 0.000468 seconds (1.93 k allocations: 219.625 KiB)
