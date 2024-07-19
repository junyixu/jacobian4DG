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

    # Trixi.init_elements!:
    nodes = Trixi.get_nodes(basis)
    # Compute the length of the 1D reference interval by integrating
    # the function with constant value unity on the corresponding
    # element data type (using \circ)
    reference_length = integrate(one ∘ eltype, nodes, basis)
    # Compute the offset of the midpoint of the 1D reference interval
    # (its difference from zero)
    reference_offset = (first(nodes) + last(nodes)) / 2

    # Store cell ids
    elements.cell_ids .= cell_ids

    # Calculate inverse Jacobian and node coordinates
    for element in eachelement(elements)
        # Get cell id
        cell_id = cell_ids[element]

        # Get cell length
        dx = Trixi.length_at_cell(mesh.tree, cell_id)

        # Calculate inverse Jacobian
        jacobian = dx / reference_length
        elements.inverse_jacobian[element] = inv(jacobian)

        # Calculate node coordinates
        # Note that the `tree_coordinates` are the midpoints of the cells.
        # Hence, we need to add an offset for `nodes` with a midpoint
        # different from zero.
        for i in eachnode(basis)
            elements.node_coordinates[1, i, element] = (mesh.tree.coordinates[1,
                                                                              cell_id] +
                                                        jacobian *
                                                        (nodes[i] - reference_offset))
        end
    end
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
