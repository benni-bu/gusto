"""
Set up and solve an example problem (purely in PETSc) that can be solved using the preconditioner 
defined in toy_mlprecon.py
"""

from firedrake.petsc import PETSc
import numpy as np
from toy_mlprecon import (ToyMLPreconditioner, Laplace1D, Jacobi)

def set_ops():
    # Set coefficients in the matrix A and RHS vector B
    A = PETSc.Mat()
    A.create(comm=PETSc.COMM_WORLD)
    A.setSizes([100,100])
    A.setType(PETSc.Mat.Type.PYTHON)
    A.setPythonContext(Laplace1D())
    A.setUp()

    # Set a right-hand side vector
    X, B = A.getVecs()
    #B.setArray(np.random.rand(10))
    B.setArray(np.ones(100))

    B_array = B.getArray()
    print("Input RHS vector B:", B_array)

    return A, X, B

def solve_linear_system(A, X, B):
    # Create a linear solver context
    ksp = PETSc.KSP().create()
    ksp.setType(PETSc.KSP.Type.GCR)
    ksp.setTolerances(rtol=1e-5)
    #ksp.setTolerances(max_it=40)

    # Set the operator (coefficient matrix) for the linear solver
    ksp.setOperators(A)

    #define the pc so PETSc knows what to do when instantiating the ML context
    pc = ksp.getPC()

    # Set the preconditioner for the linear solver
    #pc.setType(PETSc.PC.Type.NONE) #for reference runs without pc
    #pc.setType(PETSc.PC.Type.SOR)  #for reference runs with SOR
    pc.setType(PETSc.PC.Type.PYTHON)
    pc.setPythonContext(ToyMLPreconditioner())
    #pc.setPythonContext(Jacobi())

    ksp.solve(B, X)

    # Print the solution vector
    X_array = X.getArray()
    print("Solution vector X:", X_array)

    # print error norm
    r = B.duplicate()
    A.mult(X, r)
    r.aypx(-1, B)
    rnorm = r.norm()
    PETSc.Sys.Print('error norm = %g' % rnorm,
                    comm=PETSc.COMM_WORLD)

    # Free resources
    ksp.destroy()
    A.destroy()
    X.destroy()
    B.destroy()


if __name__ == "__main__":
    A, X, B = set_ops()
    solve_linear_system(A, X, B)
