#Met Office implementation
def GCR_MO(b, its, pc=None):
    if pc is not None:
        #dx = pc(b)
        dx = np.zeros(100)
    else:
        dx = np.zeros(100)
    Ax = np.dot(dx, Lapl)
    r = b-Ax

    # set vectors up for a restart value of 150
    v = np.zeros((150, 100))
    Pv = np.zeros((150, 100))

    #array to store residuals in 
    rs = np.ones((150, 100))
    
    for iv in range(its):
        if pc is not None:
            # apply the preconditioner
            Pv[iv] = pc(r)
        else:
            Pv[iv] = r
        # apply the operator
        v[iv] = np.dot(Pv[iv], Lapl)
        #print(iv)
        for ivj in range(iv):
            alpha = v[iv].dot(v[ivj])
            #print(alpha)
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
