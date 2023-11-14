"""
Set up and solve an example firedrake problem that can be solved using the preconditioner 
defined in toy_mlprecon.py
"""

from firedrake import UnitSquareMesh
from firedrake import FunctionSpace
from firedrake import Function
from firedrake.petsc import PETSc
from toy_mlprecon import ToyMLPreconditioner

#define firedrake mesh and associated function space
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

# Create Firedrake functions (vectors) associated with the FunctionSpace.
# x is the LHS solution vector, y is the known RHS forcing.
x_firedrake = Function(V)
y_firedrake = Function(V)

# Initialize PETSc
PETSc.Sys.popErrorHandler()
PETSc.Log.begin()

# Create PETSc Krylov solver
solver = PETSc.KSP().create()

