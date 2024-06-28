#! /usr/bin/env -S julia --color=yes --startup-file=no
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#=
    issue related to gloabl variable and type-stability
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
J = jacobian_ad_forward(semi);

# %%
function my_rhs!(du_ode::AbstractVector, u_ode::AbstractVector, t, mesh, equations, initial_condition, boundary_conditions, source_terms, dg, cache)
    u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
    du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)
    Trixi.rhs!(du, u, 0.0, mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache)
    return nothing
end

    t0 = zero(real(semi))
    u_ode = compute_coefficients(t0, semi)
    du_ode = similar(u_ode)

    Trixi.@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi
    dys = zeros(length(du_ode), length(du_ode))
    dy = zero(du_ode)
    dx = zero(u_ode)
    cache_zero=Enzyme.make_zero(cache)

    for i in 1:length(du_ode)
        dx[i] = 1.0
        Enzyme.autodiff(Forward, (du, u, cache)->my_rhs!(du, u, 0.0, mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache), Duplicated(du_ode, dy), Duplicated(u_ode, dx), Duplicated(cache, cache_zero)) # ok!
        dys[:, i] = dy
        dx[i] = 0.0
    end

    # return dys
# end

# foo(semi)
dys
