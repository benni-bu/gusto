"""
The Williamson 2 shallow-water test case (solid-body rotation), solved with a
discretisation of the non-linear shallow-water equations.

This uses an icosahedral mesh of the sphere, and runs a series of resolutions
to act as a convergence test.
"""

from gusto import *
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, sin, cos, pi
import sys

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24.*60.*60.
if '--running-tests' in sys.argv:
    ref_dt = {3: 3000.}
    tmax = 3000.
    ndumps = 1
else:
    # setup resolution and timestepping parameters for convergence test
    ref_dt = {3: 4000., 4: 2000., 5: 1000., 6: 500.}
    tmax = 5*day
    ndumps = 5

# setup shallow water parameters
R = 6371220.
H = 5960.
rotated_pole = (0.0, pi/3)

# setup input that doesn't change with ref level or dt
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = IcosahedralSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=2)
    x = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'BDM', 1, rotated_pole=rotated_pole)

    # Equation
    Omega = parameters.Omega
    _, lat, _ = rotated_lonlatr_coords(x, rotated_pole)
    e_lon, _, _ = rotated_lonlatr_vectors(x, rotated_pole)
    fexpr = 2*Omega*sin(lat)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr)

    # I/O
    dirname = "williamson_2_ref%s_dt%s" % (ref_level, dt)
    dumpfreq = int(tmax / (ndumps*dt))
    output = OutputParameters(
        dirname=dirname,
        dumpfreq=dumpfreq,
        dumplist_latlon=['D', 'D_error'],
        dump_nc=True,
    )

    diagnostic_fields = [RelativeVorticity(), PotentialVorticity(),
                         ShallowWaterKineticEnergy(),
                         ShallowWaterPotentialEnergy(parameters),
                         ShallowWaterPotentialEnstrophy(),
                         SteadyStateError('u'), SteadyStateError('D'),
                         MeridionalComponent('u', rotated_pole),
                         ZonalComponent('u', rotated_pole)]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transported_fields = [TrapeziumRule(domain, "u"),
                          SSPRK3(domain, "D", subcycles=2)]
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields("u")
    D0 = stepper.fields("D")
    x = SpatialCoordinate(mesh)
    u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
    uexpr = u_max*cos(lat)*e_lon
    g = parameters.g
    Dexpr = H - (R * Omega * u_max + u_max*u_max/2.0)*(sin(lat))**2/g

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)
