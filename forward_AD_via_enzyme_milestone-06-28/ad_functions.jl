function my_rhs!(du_ode::AbstractVector, u_ode::AbstractVector, t, mesh, equations, initial_condition, boundary_conditions, source_terms, dg, cache)
    u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
    du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)
    # Trixi.reset_du!(du, dg, cache)
    Trixi.rhs!(du, u, 0.0, mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache)
    return nothing
end

function gradients_ad_forward_enzyme_cache(semi)
    t0 = zero(real(semi))
    u_ode = compute_coefficients(t0, semi)
    du_ode = similar(u_ode)

    Trixi.@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache= semi
    dys = zeros(length(du_ode), length(du_ode))
    dy = zero(du_ode)
    dx = zero(u_ode)
    cache_zero=Enzyme.make_zero(cache)

    inner_func(du, u, cache) = let
        mesh = mesh
        equations = equations
        initial_condition = initial_condition
        boundary_conditions = boundary_conditions
        source_terms = source_terms
        solver = solver
        cache = cache
        my_rhs!(du, u, 0.0, mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache)
    end

    for i in 1:length(du_ode)
        dx[i] = 1.0
        Enzyme.autodiff(Forward, inner_func, Duplicated(du_ode, dy), Duplicated(u_ode, dx), Duplicated(cache, cache_zero)) # ok!
        dys[:, i] = dy
        dx[i] = 0.0
    end

    return dys
end
