"""
Rossby wave in a beta-channel. Inspired by test cases e and f in Ullrich and Jablonowski (2012).
Currently, this does not work as intended (no Rossby wave triggered by perturbation).
"""

from gusto import *
from firedrake import (PeriodicRectangleMesh, SpatialCoordinate,
                       as_vector, sin, cos, exp, pi, sqrt, min_value)
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
    ref_dt = {3: 300.}
    tmax = 3*day
    ndumps = 50

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
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, 
                                       no_normal_flow_bc_ids=[1, 2])

    # I/O
    dirname = "barocl_channel_ref%s_dt%s" % (ref_level, dt)
    dumpfreq = int(tmax / (ndumps*dt))
    output = OutputParameters(
        dirname=dirname,
        dumplist=['D', 'u'],
        dumpfreq=dumpfreq,
    )
    #diagnostic_fields = [Sum('D', 'topography')]
    #io = IO(domain, output, diagnostic_fields=diagnostic_fields)
    io = IO(domain, output)

    # Transport schemes
    transported_fields = [TrapeziumRule(domain, "u"),
                          SSPRK3(domain, "D")]
    transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

    # Time stepper
    stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods)

    # ------------------------------------------------------------------------ #
    # Initial conditions
    # ------------------------------------------------------------------------ #

    u0 = stepper.fields('u')
    D0 = stepper.fields('D')
    u_max = 35.   # Maximum amplitude of the zonal wind (m/s)
    uexpr_bg = as_vector([u_max*sin(pi*x[1]/Ly)**2, 0.0])
    g = parameters.g
    y0 = Ly/2
    Dexpr = H - u_max/(2*g) * ( (f_0-beta_0*y0) * (x[1] - Ly/(2*pi)*sin(2*pi*x[1]/Ly)) 
                                + beta_0 * (x[1]**2/2 - Ly*x[1]/(2*pi)*sin(2*pi*x[1]/Ly) 
                                            - Ly**2/(4*pi**2)*cos(2*pi*x[1]/Ly)) )
    
    #set perturbation for instability 
    Lp = 6e5 #perturbation radius
    xc = 2e6
    yc = 2.5e6
    u_pert = as_vector([10*exp(-((x[0]-xc)**2+(x[1]-yc)**2)/Lp**2), 0.0])
    uexpr = uexpr_bg + u_pert

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)
