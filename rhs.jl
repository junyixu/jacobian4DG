# MD = M \ transpose(D)
Dt = copy(transpose(D))
# D'
# MB = (M \ B)
function rhs!(du, u)
    # Reset du and flux matrix
    # du .= zero(eltype(du))
    # flux_numerical = copy(du)
    flux_numerical = zeros(size(du))
    # n_elements = size(du, 2)

    # Calculate interface and boundary fluxes, $u^* = (u^*|_{-1}, 0, ..., 0, u^*|^1)^T$
    # Since we use the flux Lax-Friedrichs from Trixi.jl, we have to pass some extra arguments.
    # Trixi.jl needs the equation we are dealing with and an additional `1`, that indicates the
    # first coordinate direction.
    equations = LinearScalarAdvectionEquation1D(1.0)
    for element in 2:n_elements-1
        # left interface
        flux_numerical[1, element] = surface_flux(u[end, element-1], u[1, element], 1, equations)
        flux_numerical[end, element-1] = flux_numerical[1, element]
        # right interface
        flux_numerical[end, element] = surface_flux(u[end, element], u[1, element+1], 1, equations)
        flux_numerical[1, element+1] = flux_numerical[end, element]
    end
    # boundary flux
    flux_numerical[1, 1] = surface_flux(u[end, end], u[1, 1], 1, equations)
    flux_numerical[end, end] = flux_numerical[1, 1]

    # Calculate surface integrals, $- M^{-1} * B * u^*$
    # println("size of du: $(size(du))")
    # println("du[2, 2]: $(du[2,2])")
    # println("du[2, 2]: $(du[2,2])")
    for element in 1:n_elements
        MB * flux_numerical[:, element]
        tmp_vec = du[:, element] .- MB * flux_numerical[:, element]
        du[1, element] = tmp_vec[1] 
        du[end, element] = tmp_vec[end] 
    end

    # Calculate volume integral, $+ M^{-1} * D^T * M * u$
    for element in 1:n_elements
        flux = u[:, element]
        du[:, element] += (M \ Dt) * M * flux
        # du[:, element] += (M \ D') * M * flux
    end

    # Apply Jacobian from mapping to reference element
    for element in 1:n_elements
        du[:, element] *= 2
    end

    return nothing
end
