#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#=
    This exmaple is taken from 
    https://trixi-framework.github.io/Trixi.jl/dev/tutorials/differentiable_programming/#Computing-the-Jacobian

    Two important APIs from Enzyme.jl:
    1. Enzyme.onehot: create a one-hot tuple
    2. Enyzme.make_zero takes a data structure and constructs a deepcopy of the data structure with all of the floats set to zero and non-differentiable types like Symbols set to their primal value.
    If Enzyme gets into such a "Mismatched activity" situation where it needs to return a differentiable data structure from a constant variable, it could try to resolve this situation by constructing a new shadow data structure, such as with Enzyme.make_zero.
    However, this still can lead to incorrect results

=#

using Trixi
using Enzyme

# %%
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
J1 = jacobian_ad_forward(semi);

function jacobian_ad_forward_enzyme(semi)
    t0 = zero(real(semi))
    u_ode = compute_coefficients(t0, semi)
    du_ode = similar(u_ode)
    dy = Tuple(zeros(size(du_ode)) for _ in 1:length(u_ode))
    dx = Enzyme.onehot(u_ode)
    tuple_semi=Tuple(Enzyme.make_zero(semi) for i=1:length(u_ode))
    Enzyme.autodiff(Enzyme.Forward, (du_ode, u_ode, semi)->Trixi.rhs!(du_ode, u_ode, semi, t0), Enzyme.BatchDuplicated(du_ode, dy), Enzyme.BatchDuplicated(u_ode, dx), BatchDuplicated(semi, tuple_semi))
    return stack(dy) # a tuple to a matrix
end

J2 = jacobian_ad_forward_enzyme(semi)

J1 == J2 # ture
