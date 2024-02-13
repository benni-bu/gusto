"""
This code sets up the matrix-free operator contexts and the preconditioners 
used in the example ML preconditioning problem. The ML preconditioner is set
up using firedrake's PCBase class. 
"""

from firedrake.preconditioners import PCBase
# from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
import torch
from PyT_tinymodels import (Dense, OneD_UNet, LinReg, smoother)
import numpy as np
import os

OptDB = PETSc.Options()

INFO = OptDB.hasName('info')

def LOG(arg):
    if INFO:
        print(arg)

# set up outputting of solve vectors for debugging purposes
datadir = '/Users/GUSTO/data/debug'  
exp = '/ml_dense' 

lapl_in_path = datadir + exp + '/lapl_invecs.txt'
lapl_out_path = datadir + exp + '/lapl_outvecs.txt'
precon_in_path = datadir + exp + '/prec_invecs.txt'
precon_out_path = datadir + exp + '/prec_outvecs.txt'

if os.path.exists(lapl_in_path):
    # If the file exists, delete its contents
    open(lapl_in_path, 'w').close()  # This truncates the file

if os.path.exists(lapl_out_path):
    open(lapl_out_path, 'w').close()

if os.path.exists(precon_in_path):
    open(precon_in_path, 'w').close()

if os.path.exists(precon_out_path):
    open(precon_out_path, 'w').close()

#--------------------------#
# define operator contexts #
#--------------------------#

# main operator matrix (taken from PETSc example: 
# https://github.com/petsc/petsc/blob/main/src/ksp/ksp/tutorials)
class Laplace1D(object):

    def create(self, A):
        LOG('Laplace1D.create()')
        M, N = A.getSize()
        assert M == N

    def destroy(self, A):
        LOG('Laplace1D.destroy()')

    def view(self, A, vw):
        LOG('Laplace1D.view()')

    def setFromOptions(self, A):
        LOG('Laplace1D.setFromOptions()')

    def setUp(self, A):
        LOG('Laplace1D.setUp()')

    def assemblyBegin(self, A, flag):
        LOG('Laplace1D.assemblyBegin()')

    def assemblyEnd(self, A, flag):
        LOG('Laplace1D.assemblyEnd()')

    def getDiagonal(self, A, d):
        LOG('Laplace1D.getDiagonal()')
        M, N = A.getSize()
        h = 1.0/(M-1)
        d.set(2.0/h**2)

    def mult(self, A, x, y):
        LOG('Laplace1D.mult()')
        M, N = A.getSize()
        xx = x.getArray(readonly=1) # to numpy array
        yy = y.getArray(readonly=0) # to numpy array
        # write input vec to file to check we're doing the right thing
        if INFO:
            with open(datadir + exp + '/lapl_invecs.txt', 'a') as file:
                np.savetxt(file, [xx], delimiter=',')
        yy[0]    =  2.0*xx[0] - xx[1]
        yy[1:-1] = - xx[:-2] + 2.0*xx[1:-1] - xx[2:]
        yy[-1]   = - xx[-2] + 2.0*xx[-1]
        h = 1.0/(M-1)
        yy *= 1.0/h**2
        if INFO:
            with open(datadir + exp + '/lapl_outvecs.txt', 'a') as file:
                np.savetxt(file, [yy], delimiter=',')

    def multTranspose(self, A, x, y):
        LOG('Laplace1D.multTranspose()')
        self.mult(A, x, y)



# simplest possible matrix-free context for testing purposes
class TestCtx():
    def __init__(self):
        pass

    def mult(self, mat, x, y):
        # y <- A x
        y.setArray(x)


# matrix-free action of ML model. Will need this when using it as a pc.
# this roughly follows the description on https://www.firedrakeproject.org/petsc-interface.html
class MLCtx():
    def __init__(self, model, smoother=None) -> None:
        #model is a loaded pytorch model including state dict.
        self.model = model
        self.smoother = smoother
    def mult(self, mat, x, y):
        LOG('MLCtx.mult()')
        #need to convert PETSc vectors to torch tensors
        x_array = x.getArray()

        if INFO:
            with open(datadir + exp + '/prec_invecs.txt', 'a') as file:
                np.savetxt(file, [x_array], delimiter=',')

        #LOG(f'PC Input vector: {x_array}')
        x_tensor = torch.tensor(x_array, dtype=torch.float32)
        # add channel dimension to make compatible with CNNs
        #x_tensor = x_tensor.unsqueeze(0)
        
        with torch.no_grad():
            y_tensor = self.model(x_tensor)
            if self.smoother is not None:
                y_tensor = self.smoother(y_tensor)

        #convert back to PETSc vector
        y_array = torch.Tensor.numpy(y_tensor)

        if INFO:
            with open(datadir + exp + '/prec_outvecs.txt', 'a') as file:
                np.savetxt(file, [y_array], delimiter=',')
        #LOG(f'PC Output vector: {y_array}')
        y.setArray(y_array)


#------------------------#
# define preconditioners #
#------------------------#

# Jacobi Preconditioner for sanity check (taken from PETSc example: 
# https://github.com/petsc/petsc/blob/main/src/ksp/ksp/tutorials)
class Jacobi(object):

    def create(self, pc):
        LOG('Jacobi.create()')
        self.diag = None

    def destroy(self, pc):
        LOG('Jacobi.destroy()')
        if self.diag:
            self.diag.destroy()

    def view(self, pc, vw):
        LOG('Jacobi.view()')

    def setFromOptions(self, pc):
        LOG('Jacobi.setFromOptions()')

    def setUp(self, pc):
        LOG('Jacobi.setUp()')
        A, B = pc.getOperators()
        self.diag = B.getDiagonal(self.diag)

    def apply(self, pc, x, y):
        LOG('Jacobi.apply()')
        if INFO:
            x_array = x.getArray()
            with open(datadir + exp + '/prec_invecs.txt', 'a') as file:
                np.savetxt(file, [x_array], delimiter=',')
        y.pointwiseDivide(x, self.diag)
        if INFO:
            y_array = y.getArray()
            with open(datadir + exp + '/prec_outvecs.txt', 'a') as file:
                np.savetxt(file, [y_array], delimiter=',')

    def applyTranspose(self, pc, x, y):
        LOG('Jacobi.applyTranspose()')
        self.apply(pc, x, y)


# actual ML preconditioner
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
        #model = OneD_UNet(1,1).to(device)
        #model.load_state_dict(torch.load("/Users/GUSTO/environments/firedrake/src/gusto/learning/unet_poisson.pth"))

        model = Dense().to(device)
        model.load_state_dict(torch.load("/Users/GUSTO/environments/firedrake/src/gusto/learning/poisson.pth"))

        #model = LinReg().to(device)
        #model.load_state_dict(torch.load("/Users/GUSTO/environments/firedrake/src/gusto/learning/lin_poisson.pth"))

        Smoother = smoother().to(device)

        #this is how P in defined in preconditioners.py as well as in the firedrake examples. But where does it get
        #the operators from? We're not passing them in when instantiating the context! - It comes from the ksp object
        # we set in the driver.
        _, P = pc.getOperators()

        #define PC operator. In Firedrake 'interfacing directly with PETSc', they just use the original matrix for 
        #this, but that's not what I want to do here.

        #build ML model matrix-free context
        Pctx = MLCtx(model, Smoother)

        #sub in test pctx for debugging purposes
        #Pctx = TestCtx()

        #Set up PETSc operator based on that context
        self.P_ml = PETSc.Mat().create()
        self.P_ml.setSizes(P.getSizes())
        self.P_ml.setType(PETSc.Mat.Type.PYTHON)
        self.P_ml.setPythonContext(Pctx)
        self.P_ml.setUp()

        #set up KSP (this mimics the vertical hybridisation PC, not sure if this is the right way to do it for me)
        #ml_pc = PETSc.PC().create()
        #ml_pc.setOptionsPrefix(prefix)
        #ml_pc.setOperators(P_ml)
        #ml_pc.setUp()
        #ml_pc.setFromOptions()
        #self.ml_pc = ml_pc
    
    def apply(self, pc, x, y):
        #with x.dat.vec_ro as x_vec, y.dat.vec as y_vec:
        # y <- ML(x)
        LOG('applying MLPC')
        self.P_ml.mult(x, y)
        #self.ml_pc.apply(x, y)    

    def applyTranspose(self, pc, X, Y):
        return super().applyTranspose(pc, X, Y)
    
    def update(self, pc):
        prefix = pc.getOptionsPrefix()
        if prefix is None:
            prefix = ""
        prefix += "toy_mlprecon_"
        pc.setOptionsPrefix(prefix)
        super().update(pc)
            
