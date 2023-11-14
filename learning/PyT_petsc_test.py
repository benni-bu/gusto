"""
This code is for getting a feel for how interfacing between PyTorch and 
PETSc might work by doing some simple PETSc operations on PyTorch
output. 
Skeleton for parts of this kindly provided by GPT3.5 :)
"""

from firedrake.petsc import PETSc
import torch
from PyT_tinymodel import TinyModel
import numpy as np

#petsc.init()
#print(torch.__version__)

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

#function to make prediction call simpler (decorated by gradient calculation disabler)
@torch.no_grad()
def predict(model, x):
  y = model(x)
  return y

#function to add two vectors in a PETSc way
def add_vectors(arr1, arr2):
    # Initialize PETSc
    PETSc.Sys.popErrorHandler()
    PETSc.Log.begin()

    # Convert NumPy arrays to PETSc vectors
    vec1 = PETSc.Vec().createWithArray(arr1)
    vec2 = PETSc.Vec().createWithArray(arr2)

    # Check if the sizes of the arrays are the same
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same size for addition")

    # Create a result vector to store the sum
    result_vec = vec1.copy()

    # Add the vectors together
    result_vec.axpy(1.0, vec2)  # result_vec = vec1 + 1.0 * vec2

    # Get the result as a NumPy array
    result_array = result_vec.getArray()

    # Destroy PETSc vectors
    vec1.destroy()
    vec2.destroy()
    result_vec.destroy()

    return result_array

#define a simple vector that we want to add our output to
stdvec = np.ones(10)

#vector that we will feed into ML model
testvec = 3*np.ones(10)
#need to convert to torch tensor first
testvec = torch.tensor(testvec, dtype=torch.float32)

#ML model inference on that vector
mloutvec = predict(model, testvec)
#convert back to np array
mloutvec = torch.Tensor.numpy(mloutvec)

#add vectors in non-PETSc way as check
check = mloutvec+stdvec

#add vectors in PETSc way
petsout = add_vectors(stdvec, mloutvec)

#print(mloutvec)
print(petsout-check)
print('Yay we can make PETSc and PyTorch talk :)')

