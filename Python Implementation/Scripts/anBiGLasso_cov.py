import numpy as np
from scipy.optimize import lsq_linear
from Scripts.utilities import K, LASSO
from Scripts.nonparanormal import nonparanormal
from Scripts.anBiGLasso import shrink, eigenvectors_MLE, calculateEigenvalues

def anBiGLasso(
    T: "(n, n) within-row covariance matrix",
    S: "(p, p) within-row covariance matrix",
    beta_1: "L1 penalty for Psi",
    beta_2: "L1 penalty for Theta",
    B_approx_iters: (int, "Hyperparameter") = 10,
):
    """
    See `calculateEigenvalues` for explanation of
    `B_approx_iters`.
    """
    n = T.shape[0]
    p = S.shape[0]
    
    if B_approx_iters > min(B_approx_iters, min(n, p)):
        # We could, and probably should, get rid of this
        # issue by randomly sampling from the true B instead
        # of following a fixed order for the approximation.
        print("Warning: B_approx_iters is too high")
        B_approx_iters = min(B_approx_iters, min(n, p))
        
    U, V = eigenvectors_MLE(T, S)
    u, v = eigenvalues_MLE(T, S, U, V, B_approx_iters)
    Psi = U @ np.diag(u) @ U.T
    Theta = V @ np.diag(v) @ V.T
    
    if beta_1 > 0:
        Psi = shrink(Psi, beta_1)
    if beta_2 > 0:
        Theta = shrink(Theta, beta_2)
    
    return Psi, Theta

def calculateSigmas(
    Xs: "(m, n, p) tensor, m samples of (n, p) matrix"
) -> "(n, p) tensor: n slices of diagonals of (n, p, p) covariance matrix":
    """
    Gets an MLE for variances of our rescaled Ys
    
    An implementation of Lemma 2
    """
    
    (m, n, p) = Xs.shape
    return (Xs**2).sum(axis=0) / m

def eigenvalues_MLE(
    T: "(n, n) empirical covariance",
    S: "(p, p) empirical covariance",
    U: "(n, n) eigenvectors of Psi",
    V: "(p, p) eigenvectors of Theta",
    B_approx_iters: int,
    for_testing_params = None
):
    """
    An implementation of Theorem 3
    
    Note: We can probably calculate this quicker
    if we use the fact that we know `u` in the
    calculation of `v` - but speed isn't an issue
    at the moment.
    """
    
    n, _ = T.shape
    p, _ = S.shape
    
    # For Psi eigenvalues
    Sigmas = U.shape[0] * np.diag(U.T @ T @ U)
    Sigmas = np.tile(Sigmas.reshape(n, 1), (1, p))
    u, _ = calculateEigenvalues(Sigmas, B_approx_iters)
    
    # For Theta eigenvalues
    Sigmas = V.shape[0] * np.diag(V.T @ S @ V)
    Sigmas = np.tile(Sigmas.reshape(1, p), (n, 1))
    _, v = calculateEigenvalues(Sigmas, B_approx_iters)
    
    
    return np.abs(u), np.abs(v)