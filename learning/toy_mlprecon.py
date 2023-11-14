"""
Here we define a 'toy' implementation of a ml preconditioner, making use of the tiny PyTorch 
model set up in PyT_tinymodel.py and using either Firedrakes PCBase class or PETSc's PCShell.
"""

from firedrake.preconditioners import PCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
import torch
from PyT_tinymodel import TinyModel
import numpy as np


# Define your custom preconditioner class
class ToyMLPreconditioner(PCBase):
    def initialize(self, pc):
        #reviev this after finishing to see what we actually need (probably only a fraction...)
        from firedrake import (FunctionSpace, Function, Constant, Cofunction,
                               FiniteElement, TensorProductElement,
                               TrialFunction, TrialFunctions, TestFunction,
                               DirichletBC, interval, MixedElement, BrokenElement)
        from firedrake.assemble import (allocate_matrix, OneFormAssembler,
                                        TwoFormAssembler)
        
        # Extract PC context
        prefix = pc.getOptionsPrefix() + "toy_mlprecon_"
        _, P = pc.getOperators()
        self.ctx = P.getPythonContext()

        #initialise PyTorch
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")

        #load PyTorch model
        model = TinyModel().to(device)
        model.load_state_dict(torch.load("tinymodel.pth"))

        #define PC operator. Not sure if this operator symbolises the action of the preconditioner or of the system to be solved.
        MLmat = 


        #set up KSP (this mimics the vertical hybridisation PC, not sure if this is the right way to do it for me)
        ml_ksp = PETSc.KSP().create(comm=pc.comm)
        ml_ksp.setOptionsPrefix(prefix)
        ml_ksp.setOperators(MLmat)
        ml_ksp.setUp()
        ml_ksp.setFromOptions()
        self.ml_ksp = ml_ksp
    
    def apply(self, pc, x, y):
        with x.dat.vec_ro as x_vec, y.dat.vec as y_vec:
            # Access the PETSc vectors from Firedrake data
            x_array = x_vec.array
            y_array = y_vec.array
            
            # Convert x_array to a PyTorch tensor
            x_torch = torch.tensor(x_array)
            
            # Perform your PyTorch-based preconditioning operation
            # For example:
            # y_torch = your_pytorch_preconditioner_function(x_torch)
            
            # Convert the result back to a PETSc vector
            y_vec.array = y_torch.numpy()  # Assuming y_torch is a PyTorch tensor
            