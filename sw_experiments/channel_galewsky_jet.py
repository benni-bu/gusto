"""
Galewsky jet in a beta-channel. Coriolis expr and general channel dimensions
from test cases e and f in Ullrich and Jablonowski (2012). 
I don't use the exact same velocity field as Galewsky et al. because I can't be 
bothered doing numerical integration for the height field, so I use a 
slightly different but similar Gaussian shape instead. This is okay - I do 
not need to worry about discontinuities in the first derivative at the poles as I 
have no poles.
"""

from gusto import *
from firedrake import (PeriodicRectangleMesh, SpatialCoordinate,
                       as_vector, sin, cos, exp, erf, pi, sqrt, min_value)
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
    tmax = 0.5*day
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
    y0 = Ly/2 
    fexpr = f_0 + beta_0*(x[1]-y0)
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
    u_max = 80.   # Maximum amplitude of the zonal wind (m/s)
    #ys = Ly * 26/90 # southern and northern boundaries of jet chosen to be similar to Galewsky
    #yn = Ly - Ly * 26/90
    #u_mask = lambda y, ylower, yupper: 1 if ylower < y < yupper else 0
    s = 5e5 #~width of jet in m. Chosen to give a jet similar to Galewsky
    uexpr = as_vector([u_max * exp(-((x[1]-y0)/s)**2), 0.0])
    g = parameters.g
    Dexpr_bg = H - s * u_max/(2*g) *  ( beta_0 * s * exp(((y0 - x[1])/s)**2)
                                - sqrt(pi) * f_0 * erf((y0 - x[1])/s))
    
    #set perturbation for instability 
    Lp = 6e5 #perturbation radius
    xc = 2e6
    D_pert = 120*exp(-((x[0]-xc)**2+(x[1]-y0)**2)/Lp**2)
    Dexpr = Dexpr_bg + D_pert

    u0.project(uexpr)
    D0.interpolate(Dexpr)

    Dbar = Function(D0.function_space()).assign(H)
    stepper.set_reference_profiles([('D', Dbar)])

    # ------------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------------ #

    stepper.run(t=0, tmax=tmax)
