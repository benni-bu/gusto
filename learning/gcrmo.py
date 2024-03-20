#Met Office implementation
def GCR_MO(b, its, pc=True):
    if pc == True:
        #dx = prec(b)
        dx = np.zeros(100)
    else:
        dx = np.zeros(100)
    Ax = np.dot(dx, Lapl)
    r = b-Ax

    # set vectors up for a restart value of 3000
    v = np.zeros((3000, 100))
    Pv = np.zeros((3000, 100))

    #array to store residuals in 
    rs = np.ones((3000, 100))
    
    for iv in range(its):
        if pc == True:
            # apply the preconditioner
            #Pv[iv] = prec(r)
            #Pv[iv] = Jacobi(r)
            Pv[iv] = FabPC(r)
        else:
            Pv[iv] = r
        # apply the operator
        v[iv] = np.dot(Pv[iv], Lapl)
        for ivj in range(iv+1):
            alpha = v[iv].dot(v[ivj])
            v[iv] += -alpha * v[ivj]
            Pv[iv] += -alpha * Pv[ivj]
        alpha = np.linalg.norm(v[iv])
        beta = 1.0 / alpha
        v[iv] *= beta
        Pv[iv] *= beta
        alpha = r.dot(v[iv])
        dx += alpha * Pv[iv]
        r += -alpha * v[iv]
        rs[iv] = r
    return dx, rs, Pv, v