"""
This is a python wrapper for scBiGLasso
"""

import matlab.engine
import io
import numpy as np
from Scripts.anBiGLasso import calculate_empirical_covariances

eng = matlab.engine.start_matlab()
# It will be obvious which path you need, as matlab.engine is
# thankfully smart enough to tell you!  Try running code with
# `eng` and if it fails, it should tell you the path to put
# in this function.
# TODO: consider eng.cd(...) instead
eng.addpath(
    '/Users/baileyandrew/Desktop/Python Notebooks.nosync/Research/scBiGLasso '
    + 'Implementation/Scalable_Bigraphical_Lasso'
)

def scBiGLasso(Ys, beta_1, beta_2):
    T, S = calculate_empirical_covariances(Ys)
    T_matlab = matlab.double(T)
    S_matlab = matlab.double(S)
    b_matlab = matlab.double(np.array([beta_1, beta_2]))
    
    _, _, Psi_matlab, Theta_matlab = eng.scBiGLasso(
        S_matlab,
        T_matlab,
        b_matlab
        nargout=4,
        stdout=io.StringIO()
    )
    
    Psi = np.asarray(Psi_matlab)
    Theta = np.asarray(Theta_matlab)
    
    return Psi, Theta
    