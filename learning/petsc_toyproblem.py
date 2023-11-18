"""
Set up and solve an example problem (purely in PETSc) that can be solved using the preconditioner 
defined in toy_mlprecon.py
"""

from firedrake.petsc import PETSc
import numpy as np
from toy_mlprecon import ToyMLPreconditioner

def create_matrix_and_vectors():
    # Initialize PETSc
    PETSc.Sys.popErrorHandler()  # Suppresses error handler to avoid MPI initialization warning
    PETSc.Log.begin()

    # Create a matrix A
    A = PETSc.Mat().create()
    A.setSizes([10, 10])  # Adjust the size according to your problem
    #A.setType(PETSc.Mat.Type.MATSEQAIJ)  # Sequential AIJ format
    A.setFromOptions()
    A.setUp()

    # Create vectors X (solution) and B (right-hand side)
    X, B = A.createVecs()

    return A, X, B

def set_coefficients(A, X, B):
    # Set coefficients in the matrix A and vectors X, B
    A.assemble()
    A.setDiagonal(PETSc.Vec().createWithArray(2*np.ones(10)))

    # Set a right-hand side vector
    B.setArray(PETSc.Vec().createWithArray(np.random.rand(10)))

def solve_linear_system(A, X, B):
    # Create a linear solver context
    ksp = PETSc.KSP().create()

    # Set the operator (coefficient matrix) for the linear solver
    ksp.setOperators(A)

    # Create an instance of your preconditioner
    ml_precon = ToyMLPreconditioner()

    # Set the preconditioner for the linear solver
    ksp.getPC().setType(PETSc.PC.Type.PYTHON)
    ksp.getPC().setPythonContext(ml_precon)

    ksp.solve(B, X)

    # Print the solution vector
    X_array = X.getArray()
    print("Solution vector X:", X_array)

    # Free resources
    ksp.destroy()
    A.destroy()
    X.destroy()
    B.destroy()

    # Finalize PETSc
    PETSc.finalize()

if __name__ == "__main__":
    A, X, B = create_matrix_and_vectors()
    set_coefficients(A, X, B)
    solve_linear_system(A, X, B)
