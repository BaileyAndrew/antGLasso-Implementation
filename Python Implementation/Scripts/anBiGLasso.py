import numpy as np
from Scripts.utilities import K, LASSO

def anBiGLasso(
    Ys: "(m, n, p) input tensor",
    beta_1: "L1 penalty for Psi",
    beta_2: "L1 penalty for Theta"
):
    (m, n, p) = Ys.shape
    T = np.einsum("mnp, mlp -> nl", Ys, Ys) / (m*p)
    S = np.einsum("mnp, mnl -> pl", Ys, Ys) / (m*n)
    U, V = eigenvectors_MLE(T, S)
    u, v = eigenvalues_MLE(Ys, U, V)
    Psi = U @ np.diag(u) @ U.T
    Theta = V @ np.diag(v) @ V.T
    
    if beta_1 > 0:
        Psi = shrink(Psi, beta_1)
    if beta_2 > 0:
        Theta = shrink(Theta, beta_2)
    
    return Psi, Theta

def shrink(
    Psi: "Matrix to shrink row by row",
    b: "L1 penalty per row"
) -> "L1-shrunk Psi":
    n = Psi.shape[0]
    for r in range(n):
        row = np.delete(Psi[r, :], r, axis=0)
        row = LASSO(np.eye(n-1), row, b)
        Psi[r, :r] = row[:r]
        Psi[r, r+1:] = row[r:]
        Psi[:, r] = Psi[r, :]
    return Psi

def eigenvectors_MLE(
    T: "Within-row empirical covariance matrix",
    S: "Within-column empirical covariance matrix"
) -> "Tuple of the MLE estimates of eigenvectors of Psi/Theta":
    """
    An implementation of Theorem 1
    """
    n = T.shape[0]
    p = S.shape[0]
    
    _T = T * K(n, 2*p-1, p)
    _S = S * K(p, 2*n-1, n)
    
    u, U = np.linalg.eigh(_T)
    v, V = np.linalg.eigh(_S)
    
    return U, V

def rescaleYs(
    Ys: "(m, n, p) tensor, m samples of (n, p) matrix",
    U: "(n, n) eigenvectors of Psi",
    V: "(p, p) eigenvectors of Theta"
) -> "(m, p, n) tensor":
    """
    Rescales our input data to be drawn from a kronecker sum
    distribution with parameters being the eigenvalues
    
    An implementation of Lemma 1
    """
    return U.T @ Ys @ V

def calculateSigmas(
    Xs: "(m, n, p) tensor, m samples of (n, p) matrix"
) -> "(n, p) tensor: n slices of diagonals of (n, p, p) covariance matrix":
    """
    Gets an MLE for variances of our rescaled Ys
    
    An implementation of Lemma 2
    """
    
    (m, n, p) = Xs.shape
    return (Xs**2).sum(axis=0) / m

def calculateEigenvalues(
    Sigmas: "(n, p) tensor",
):
    """
    Solves system of linear equations for the eigenvalues
    
    An implementation of Lemma 3
    """
    
    (n, p) = Sigmas.shape
    invSigs = 1 / Sigmas
    
    
    a = invSigs.T.reshape((n*p,))
    B = np.empty((
        n * p, n + p 
    ))
    for row in range(n*p):
        i = row % n
        j = row // n
        B[row, :] = 0
        B[row, i] = 1
        B[row, n+j] = 1
    
    Ls = np.linalg.lstsq(B, a, rcond=None)[0]
    print(Ls)
    Ls = np.zeros((n+p,))
    iters = 10
    for i in range(iters):
        B_ = np.concatenate(
            [
                B[i*n:(i+1)*n],
                B[i::n]
            ]
        )
        a_ = np.concatenate(
            [
                a[i*n:(i+1)*n],
                a[i::n]
            ]
        )
        Ls += np.linalg.lstsq(B_, a_, rcond=None)[0]
    Ls /= iters
    print(Ls)
    #print(B_)
    
    
    
    #print(Ls)
    #Ls = np.linalg.lstsq(B_, a_, rcond=None)[0]
    #print(Ls)
    
    return Ls[:n], Ls[n:]

def eigenvalues_MLE(
    Ys: "(m, n, p) tensor",
    U: "(n, n) eigenvectors of Psi",
    V: "(p, p) eigenvectors of Theta",
):
    """
    An implementation of Theorem 2
    """
    Xs = rescaleYs(Ys, U, V)
    Sigmas = calculateSigmas(Xs)
    u, v = calculateEigenvalues(Sigmas)
    return u, v