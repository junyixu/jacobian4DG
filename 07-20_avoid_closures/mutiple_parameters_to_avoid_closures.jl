using Trixi
using Enzyme
using Polyester:@batch

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
function my_rhs!(du_ode::AbstractVector, u_ode::AbstractVector, mesh, equations, initial_condition, boundary_conditions, source_terms, solver, boundaries, _node_coordinates, cell_ids, node_coordinates, inverse_jacobian, _neighbor_ids, neighbor_ids, orientation, surface_flux_values, u)

    # RealT = real(solver.basis)
    # uEltype = RealT
    # elements = Trixi.ElementContainer1D{RealT, uEltype}(inverse_jacobian,node_coordinates,surface_flux_values,cell_ids,_node_coordinates,vec(surface_flux_values))
    elements = Trixi.ElementContainer1D(inverse_jacobian,node_coordinates,surface_flux_values,cell_ids,_node_coordinates,vec(surface_flux_values))
    # interfaces = Trixi.InterfaceContainer1D{uEltype}(u, neighbor_ids, orientation, vec(u), _neighbor_ids)
    interfaces = Trixi.InterfaceContainer1D(u, neighbor_ids, orientation, vec(u), _neighbor_ids)

    cache = (; boundaries, elements, interfaces)

    u_wrap = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
    du_wrap = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

    Trixi.rhs!(du_wrap, u_wrap, 0.0, mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache)
    return nothing
end

# %%

function gradient_ad_forward_enzyme_cache(semi)
    t0 = zero(real(semi))
    u_ode = compute_coefficients(t0, semi)
    du_ode = similar(u_ode)

    # dxs = zeros(length(du_ode), length(du_ode))
    dy = zero(du_ode)
    dx = zero(u_ode)

    Trixi.@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi
    (; boundaries, elements, interfaces) = cache

    dx[1] = 1.0
    Enzyme.autodiff(Forward, my_rhs!, Duplicated(du_ode, dy), Duplicated(u_ode, dx), Const(mesh), Const(equations), Const(initial_condition), Const(boundary_conditions), Const(source_terms), Const(solver), Const(boundaries),
    Const(elements._node_coordinates),
    Const(elements.cell_ids),
    Const(elements.node_coordinates),
    Const(elements.inverse_jacobian),
    Const(interfaces._neighbor_ids),
    Const(interfaces.neighbor_ids),
    Const(interfaces.orientations),
    Duplicated(elements.surface_flux_values, Enzyme.make_zero(elements.surface_flux_values)),
    Duplicated(interfaces.u, Enzyme.make_zero(interfaces.u)))

# my_rhs!(du_ode, u_ode, mesh, equations, initial_condition, boundary_conditions, source_terms, solver, boundaries, elements._node_coordinates, elements.cell_ids, elements.node_coordinates, elements.inverse_jacobian, interfaces._neighbor_ids, interfaces.neighbor_ids, interfaces.orientations, elements.surface_flux_values, interfaces.u)
    return dy
end

gradient_ad_forward_enzyme_cache(semi);
