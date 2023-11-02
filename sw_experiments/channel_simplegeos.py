"""
Steady Geostrophic flow in a periodic channel domain. 
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
    ndumps = 10

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
    eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr,
                                       no_normal_flow_bc_ids=[1, 2])

    # I/O
    dirname = "linear_channel_simplegeos_pctest_ref%s_dt%s" % (ref_level, dt)
    dumpfreq = int(tmax / (ndumps*dt))
    output = OutputParameters(
        dirname=dirname,
        dumplist=['D', 'u'],
        dumpfreq=dumpfreq,
    )
    io = IO(domain, output)

    # Transport schemes
    transport_schemes = [ForwardEuler(domain, "D")]
    transport_methods = [DefaultTransport(eqns, "D")]

    # solver
    alpha = 0.5
    linear_solver = MLLinearSolver(eqns, alpha)

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transport_schemes, transport_methods, linear_solver=linear_solver)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields('u')
    D0 = stepper.fields('D')
    u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
    uexpr = as_vector([u_max, 0.0])
    g = parameters.g
    Dexpr = H - u_max/g*((f_0-beta_0*Ly/2)*x[1] + beta_0/2*x[1]**2)

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)
