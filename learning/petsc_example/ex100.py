'''
PETSc tutorial taken from https://github.com/petsc/petsc/blob/main/src/ksp/ksp/tutorials
'''


def RunTest():

    from petsc4py import PETSc
    import example100

    OptDB = PETSc.Options()
    N     = OptDB.getInt('N', 100)
    draw  = OptDB.getBool('draw', False)

    A = PETSc.Mat()
    A.create(comm=PETSc.COMM_WORLD)
    A.setSizes([N,N])
    A.setType(PETSc.Mat.Type.PYTHON)
    A.setPythonContext(example100.Laplace1D())
    A.setUp()

    x, b = A.getVecs()
    b.set(1)

    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    ksp.setType(PETSc.KSP.Type.PYTHON)
    ksp.setPythonContext(example100.ConjGrad())
    #ksp.setType(PETSc.KSP.Type.CG)
    ksp.setTolerances(max_it=40)

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.NONE) #for reference runs without pc
    #pc.setType(PETSc.PC.Type.PYTHON)
    #pc.setPythonContext(example100.Jacobi())

    ksp.setOperators(A, A)
    ksp.setFromOptions()
    ksp.solve(b, x)

    r = b.duplicate()
    A.mult(x, r)
    r.aypx(-1, b)
    rnorm = r.norm()
    PETSc.Sys.Print('error norm = %g' % rnorm,
                    comm=PETSc.COMM_WORLD)

    if draw:
        viewer = PETSc.Viewer.DRAW(x.getComm())
        x.view(viewer)
        PETSc.Sys.sleep(2)

if __name__ == '__main__':
    import sys, petsc4py
    petsc4py.init(sys.argv)
    RunTest()

