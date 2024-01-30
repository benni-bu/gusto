from gusto import *
from firedrake import (PeriodicRectangleMesh, SpatialCoordinate, project,
                       as_vector, sin, cos, exp, erf, pi, sqrt)

dt = 10

mesh = PeriodicRectangleMesh(10, 10, 100, 100, "x", quadrilateral=True)

x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'RTCF', 1)

parameters = ShallowWaterParameters(H = 5000)

R = 6371220.
Omega = parameters.Omega

beta_0 = 2/R*(Omega*cos(pi/4))
f_0 = 2*Omega*sin(pi/4)
y0 = 50 
fexpr = f_0 + beta_0*(x[1]-y0)

eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr)

uexpr = as_vector([100 * exp(-((x[1]-50)/10)**2), 0.0])

u0 = eqns.fields('u')
u0.project(uexpr, 'DG')

y = erf(1)
print(y)