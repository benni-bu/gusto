"""
This aims to set up an equivalent to Williamson 5 in a periodic channel domain. 
solved with a discretisation of the linear shallow-water equations.
"""

from gusto import *
from firedrake import (PeriodicRectangleMesh, SpatialCoordinate,
                       as_vector, sin, cos, pi, sqrt, min_value)
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
    ref_dt = {1: 1500.}
    tmax = 1*day
    ndumps = 5

#setup domain parameters
Lx = 4.0e7  # length
Ly = 6.0e6  # width
deltax = 2.5e5
deltay = deltax

# setup shallow water parameters
R = 6371220. #Earth's radius only defined for the purpose of calculating Coriolis parameter
H = 5960.

# setup input that doesn't change with ref level or dt
parameters = ShallowWaterParameters(H=H)

for ref_level, dt in ref_dt.items():

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    # Domain
    ncolumnsx = int(ref_level*Lx/deltax)
    ncolumnsy = int(ref_level*Ly/deltay)  
    mesh = PeriodicRectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly, "x", quadrilateral=True)
    x = SpatialCoordinate(mesh)
    domain = Domain(mesh, dt, 'RTCF', 1)

    # Equation
    Omega = parameters.Omega
    #see Ullrich&Jablonowski 2012 for reference on beta-plane expression
    beta_0 = 2/R*(Omega*cos(pi/4))
    f_0 = 2*Omega*sin(pi/4)
    fexpr = f_0 + beta_0*(x[1]-Ly/2)
    R0 = 1e6 
    R0sq = R0**2
    y_c = 3e6 #place mountain in the centre of the channel
    ysq = (x[1] - y_c)**2
    x_c = 5e6
    xsq = (x[0] - x_c)**2
    rsq = min_value(R0sq, ysq+xsq)
    r = sqrt(rsq)
    bexpr = 2000 * (1 - r/R0)
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr,
                                       no_normal_flow_bc_ids=[1, 2])

    # I/O
    dirname = "linear_w5_channel_ref%s_dt%s" % (ref_level, dt)
    dumpfreq = int(tmax / (ndumps*dt))
    output = OutputParameters(
        dirname=dirname,
        dumplist=['D', 'u'],
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
    uexpr = as_vector([u_max, 0.0])
    g = parameters.g
    Dexpr = H - u_max/g*((f_0-beta_0*Ly/2)*x[1] + beta_0/2*x[1]**2) - bexpr

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)
