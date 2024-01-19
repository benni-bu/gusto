from gusto import *
from firedrake import (PeriodicRectangleMesh, SpatialCoordinate, project,
                       as_vector, sin, cos, exp, erf, pi, sqrt, min_value)
import sys


mesh = PeriodicRectangleMesh(10,10, 100, 100, "x", quadrilateral=True)
x = SpatialCoordinate(mesh)

uexpr = as_vector([100 * exp(-((x[1]-50)/10)**2), 0.0])

u = project(uexpr, 'DG')

y = erf(1)
print(y)