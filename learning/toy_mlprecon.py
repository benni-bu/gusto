"""
Here we define a 'toy' implementation of a ml preconditioner, making use of the tiny PyTorch 
model set up in PyT_tinymodel.py and using either Firedrakes PCBase class or PETSc's PCShell.
"""

from firedrake.preconditioners import PCBase
# from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
import torch
from PyT_tinymodel import TinyModel

OptDB = PETSc.Options()

INFO = OptDB.hasName('info')

def LOG(arg):
    if INFO:
        print(arg)

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
        yy[0]    =  2.0*xx[0] - xx[1]
        yy[1:-1] = - xx[:-2] + 2.0*xx[1:-1] - xx[2:]
        yy[-1]   = - xx[-2] + 2.0*xx[-1]
        h = 1.0/(M-1)
        yy *= 1.0/h**2

    def multTranspose(self, A, x, y):
        LOG('Laplace1D.multTranspose()')
        self.mult(A, x, y)



# simplest possible matrix-free context for testing purposes
class TestCtx():
    def __init__(self):
        pass

    def mult(self, mat, x, y):
        # y <- A x
        y = x


# matrix-free action of ML model. Will need this when using it as a pc.
# this roughly follows the description on https://www.firedrakeproject.org/petsc-interface.html
class MLCtx():
    def __init__(self, model) -> None:
        #model is a loaded pytorch model including state dict.
        self.model = model
    def mult(self, mat, x, y):
        LOG('applying MLPC')
        #need to convert PETSc vectors to torch tensors
        x_array = x.getArray()
        x_tensor = torch.tensor(x_array, dtype=torch.float32)
        
        with torch.no_grad():
            y_tensor = self.model(x_tensor)

        #convert back to PETSc vectors
        x_array = torch.Tensor.numpy(x_tensor)
        y_array = torch.Tensor.numpy(y_tensor)
        x = PETSc.Vec().createWithArray(x_array)
        y = PETSc.Vec().createWithArray(y_array)


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
        y.pointwiseDivide(x, self.diag)

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
        model = TinyModel().to(device)
        model.load_state_dict(torch.load("/Users/GUSTO/environments/firedrake/src/gusto/learning/tinymodel.pth"))

        #this is how P in defined in preconditioners.py as well as in the firedrake examples. But where does it get
        #the operators from? We're not passing them in when instantiating the context! - It comes from the ksp object
        # we set in the driver.
        _, P = pc.getOperators()

        #define PC operator. In Firedrake 'interfacing directly with PETSc', they just use the original matrix for 
        #this, but that's not what I want to do here.

        #build ML model matrix-free context
        Pctx = MLCtx(model)

        #sub in test pctx for debugging purposes
        #Pctx = TestCtx()

        #Set up PETSc operator based on that context
        
        P.setType(PETSc.Mat.Type.PYTHON)
        P.setPythonContext(Pctx)
        P.setUp()

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
            