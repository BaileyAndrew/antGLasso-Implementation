"""
This is a python wrapper for EiGLasso
"""

import matlab.engine
import io
import numpy as np

eng = matlab.engine.start_matlab()
# It will be obvious which path you need, as matlab.engine is
# thankfully smart enough to tell you!  Try running code with
# `eng` and if it fails, it should tell you the path to put
# in this function.
eng.addpath(
    '/Users/baileyandrew/Desktop/Python Notebooks.nosync/Research/scBiGLasso '
    + 'Implementation/EiGLasso/EiGLasso_JMLR'
)

def EiGLasso(Ys, beta_1=0.01, beta_2=0.01):
    (m, n, p) = Ys.shape
    T = np.einsum("mnp, mlp -> nl", Ys, Ys) / (m*p)
    S = np.einsum("mnp, mnl -> pl", Ys, Ys) / (m*n)
    T_matlab = matlab.double(T)
    S_matlab = matlab.double(S)
    beta_1_matlab = matlab.double(beta_1)
    beta_2_matlab = matlab.double(beta_2)
    Theta, Psi, ts, fs = eng.eiglasso_joint(
        S_matlab,
        T_matlab,
        beta_1_matlab,
        beta_2_matlab,
        nargout=4,
        stdout=io.StringIO()
    )
    Theta = np.asarray(Theta)
    Psi = np.asarray(Psi)
    return Psi, Theta