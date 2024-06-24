# semi.boundary_conditions |> typeof
semi.cache[:solver]

semi.cache[:elements].inverse_jacobian # Trixi.ElementContainer1D
semi.cache[:interfaces] # Trixi.InterfaceContainer1D
semi.cache[:boundaries] # Trixi.BoundaryContainer1D

ElementContainer1D

cache_zero[:elements].inverse_jacobian # Trixi.ElementContainer1D
cache_zero[:interfaces] # Trixi.InterfaceContainer1D
cache_zero[:boundaries] # Trixi.BoundaryContainer1D


SurfaceIntegralWeakForm
solver.basis == basis
solver.surface_integral.surface_flux # 似乎是个函数
solver.surface_integral.surface_flux
solver.volume_integral |> typeof
VolumeIntegralWeakForm
FluxLaxFriedrichs

