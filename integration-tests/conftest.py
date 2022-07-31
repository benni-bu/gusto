"""
This provides standard configurations for transport tests.
"""

from firedrake import (IcosahedralSphereMesh, PeriodicIntervalMesh,
                       ExtrudedMesh, SpatialCoordinate, as_vector,
                       sqrt, exp, pi)
from gusto import *
from collections import namedtuple
import pytest

opts = ('state', 'tmax', 'f_init', 'f_end', 'family', 'degree', 'uexpr', 'tol')
TracerSetup = namedtuple('TracerSetup', opts)
TracerSetup.__new__.__defaults__ = (None,)*len(opts)


def tracer_sphere(tmpdir, degree):
    mesh = IcosahedralSphereMesh(radius=1,
                                 refinement_level=3,
                                 degree=1)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    # Parameters chosen so that dt != 1
    # Gaussian is translated from (lon=pi/2, lat=0) to (lon=0, lat=0)
    # to demonstrate that transport is working correctly

    dt = pi/3. * 0.02
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=15)
    state = State(mesh, dt=dt, output=output)

    uexpr = as_vector([-x[1], x[0], 0.0])

    tmax = pi/2
    f_init = exp(-x[2]**2 - x[0]**2)
    f_end = exp(-x[2]**2 - x[1]**2)

    tol = 0.05

    return TracerSetup(state, tmax, f_init, f_end, "BDM", degree, uexpr, tol)


def tracer_slice(tmpdir, degree):
    n = 30 if degree == 0 else 15
    m = PeriodicIntervalMesh(n, 1.)
    mesh = ExtrudedMesh(m, layers=n, layer_height=1./n)

    # Parameters chosen so that dt != 1 and u != 1
    # Gaussian is translated by 1.5 times width of domain to demonstrate
    # that transport is working correctly

    dt = 0.01
    tmax = 0.75
    output = OutputParameters(dirname=str(tmpdir), dumpfreq=25)
    state = State(mesh, dt=dt, output=output)

    uexpr = as_vector([2.0, 0.0])

    x = SpatialCoordinate(mesh)
    width = 1./10.
    f0 = 0.5
    fmax = 2.0
    xc_init = 0.25
    xc_final = 0.75
    r_init = sqrt((x[0]-xc_init)**2 + (x[1]-0.5)**2)
    r_final = sqrt((x[0]-xc_final)**2 + (x[1]-0.5)**2)
    f_init = f0 + (fmax - f0) * exp(-(r_init / width)**2)
    f_end = f0 + (fmax - f0) * exp(-(r_final / width)**2)

    tol = 0.12

    return TracerSetup(state, tmax, f_init, f_end, "CG", degree, uexpr, tol)


def tracer_blob_slice(tmpdir, degree):
    dt = 0.01
    L = 10.
    m = PeriodicIntervalMesh(10, L)
    mesh = ExtrudedMesh(m, layers=10, layer_height=1.)

    output = OutputParameters(dirname=str(tmpdir), dumpfreq=25)
    state = State(mesh, dt=dt, output=output)

    tmax = 1.
    x = SpatialCoordinate(mesh)
    f_init = exp(-((x[0]-0.5*L)**2 + (x[1]-0.5*L)**2))

    return TracerSetup(state, tmax, f_init, family="CG", degree=degree)


@pytest.fixture()
def tracer_setup():

    def _tracer_setup(tmpdir, geometry, blob=False, degree=1):
        if geometry == "sphere":
            assert not blob
            return tracer_sphere(tmpdir, degree)
        elif geometry == "slice":
            if blob:
                return tracer_blob_slice(tmpdir, degree)
            else:
                return tracer_slice(tmpdir, degree)

    return _tracer_setup
