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

function my_rhs!(du_ode::AbstractVector, u_ode::AbstractVector, semi)
    Trixi.@unpack mesh, equations, initial_condition,  solver = semi

    cell_ids = Trixi.local_leaf_cells(mesh.tree)

    RealT = real(solver.basis)
    uEltype = RealT
    n_elements = length(cell_ids)
    elements = Trixi.ElementContainer1D{RealT, uEltype}(n_elements, nvariables(equations),
                                                  nnodes(solver.basis))
    basis = solver.basis

    Trixi.init_elements!(elements, cell_ids, mesh, basis)
    return nothing
end

function gradient_ad_reverse_enzyme_cache(semi)
    t0 = zero(real(semi))
    u_ode = compute_coefficients(t0, semi)
    du_ode = similar(u_ode)

    dxs = zeros(length(du_ode), length(du_ode))
    dy = zero(du_ode)
    dx = zero(u_ode)

    dy[1] = 1.0
    Enzyme.autodiff(Reverse, my_rhs!, Duplicated(du_ode, dy), Duplicated(u_ode, dx), Const(semi))
    return dx
end

gradient_ad_reverse_enzyme_cache(semi);
