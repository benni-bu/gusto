"""
Here we define a 'toy' implementation of a ml preconditioner, making use of the tiny PyTorch 
model set up in PyT_tinymodel.py and using either Firedrakes PCBase class or PETSc's PCShell.
"""

from firedrake.preconditioners import PCBase
# from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
import torch
from PyT_tinymodel import (TinyModel, MatrixFreeApp)

class TestCtx():
    def __init__(self):
        pass

    def mult(self, mat, x, y):
        # y <- A x
        y = x

class ToyMLPreconditioner(PCBase):
    def initialize(self, pc):
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
        model.load_state_dict(torch.load("/Users/GUSTO/environments/firedrake/src/gusto/learning/tinymodel.pth"))

        #this is how P in defined in preconditioners.py as well as in the firedrake examples. But where does it get
        #the operators from? We're not passing them in when instantiating the context!
        _, P = pc.getOperators()

        #define PC operator. In Firedrake 'interfacing directly with PETSc', they just use the original matrix for 
        #this, but that's not what I want to do here.

        #build ML model matrix-free context
        Pctx = MatrixFreeApp(model)

        #sub in test pctx for debugging purposes
        #Pctx = TestCtx()

        #Set up PETSc operator based on that context
        
        P.setPythonContext(Pctx)
        P.SetUp()
        P.setType(P.type.PYTHON)

        #can I maybe just forego this and tell the model to do inference in the apply function?

        #set up KSP (this mimics the vertical hybridisation PC, not sure if this is the right way to do it for me)
        ml_pc = PETSc.PC().create()
        #ml_pc.setOptionsPrefix(prefix)
        ml_pc.setOperators(P)
        ml_pc.setUp()
        ml_pc.setFromOptions()
        self.ml_pc = ml_pc
    
    def apply(self, pc, x, y):
        #with x.dat.vec_ro as x_vec, y.dat.vec as y_vec:
        # y <- A^{-1}x
        self.ml_pc.apply(x, y)    

    def applyTranspose(self, pc, X, Y):
        return super().applyTranspose(pc, X, Y)
    
    def update(self, pc):
        prefix = pc.getOptionsPrefix()
        if prefix is None:
            prefix = ""
        prefix += "toy_mlprecon_"
        pc.setOptionsPrefix(prefix)
        super().update(pc)
            