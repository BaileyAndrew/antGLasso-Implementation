"""
This is a python wrapper for TeraLasso
"""

import matlab.engine
import io
import numpy as np

eng = matlab.engine.start_matlab()
# It will be obvious which path you need, as matlab.engine is
# thankfully smart enough to tell you!  Try running code with
# `eng` and if it fails, it should tell you the path to put
# in this function.
# TODO: consider eng.cd(...) instead
eng.addpath(
    '/Users/baileyandrew/Desktop/Python Notebooks.nosync/Research/scBiGLasso '
    + 'Implementation/TeraLasso'
)

def nmode_gram(A, n):
    return np.tensordot(A, A, axes=(n, n))

def TeraLasso(Ys, betas):
    n, *d = Ys.shape
    
    Ss = []
    
    for idx, val in enumerate(d):
        gram = np.zeros((val, val))
        for i in range(n):
            gram += nmode_gram(Ys[i], -(idx+1))
        Ss.append(matlab.double(gram / n))
    
    d_matlab = matlab.double(d)
    betas_matlab = matlab.double(betas)

    Psis_matlab = eng.teralasso(
        Ss,
        d_matlab,
        'L1',
        0,
        1e-8,
        betas_matlab,
        100,
        nargout=1,
        stdout=io.StringIO()
    )
    
    Psis = []
    
    for Psi in Psis_matlab:
        Psis.append(np.asarray(Psi))
    
    return Psis