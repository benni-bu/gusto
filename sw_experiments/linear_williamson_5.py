"""
The Williamson 5 shallow-water test case (flow over topography), solved with a
discretisation of the linear shallow-water equations.

This uses an cubed mesh of the sphere.
"""

from gusto import *
from firedrake import (CubedSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value)
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
    # setup resolution and timestepping parameters
    ref_dt = {4: 1500.}
    tmax = 1*day
    ndumps = 5

# setup shallow water parameters
R = 6371220.
H = 5960.

# setup input that doesn't change with ref level or dt
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    mesh = CubedSphereMesh(radius=R,
                                 refinement_level=ref_level, degree=2)
    x = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'RTCF', 1)

    # Equation
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R
    lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    lsq = (lamda - lamda_c)**2
    theta_c = pi/6.
    thsq = (theta - theta_c)**2
    rsq = min_value(R0sq, lsq+thsq)
    r = sqrt(rsq)
    bexpr = 2000 * (1 - r/R0)
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr)

    # I/O
    dirname = "williamson_5_ref%s_dt%s" % (ref_level, dt)
    dumpfreq = int(tmax / (ndumps*dt))
    output = OutputParameters(
        dirname=dirname,
        dumplist_latlon=['D'],
        dumpfreq=dumpfreq,
    )
    diagnostic_fields = [Sum('D', 'topography')]
    io = IO(domain, output, diagnostic_fields=diagnostic_fields)

    # Transport schemes
    transport_schemes = [ForwardEuler(domain, "D")]
    transport_methods = [DefaultTransport(eqns, "D")]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transport_schemes, transport_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields('u')
    D0 = stepper.fields('D')
    u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
    g = parameters.g
    Rsq = R**2
    Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)
