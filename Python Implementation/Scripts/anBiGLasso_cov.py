import numpy as np
from Scripts.utilities import K, LASSO
from Scripts.nonparanormal import nonparanormal

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

def calculate_empirical_covariances(
    Ys: "(m, n, p)"
) -> ("(n, n), (p, p)"):
    """
    Equivalent to:
    T = np.einsum("mnp, mlp -> nl", Ys, Ys) / (m*p)
    S = np.einsum("mnp, mnl -> pl", Ys, Ys) / (m*n)
    but faster
    """
    m, n, p = Ys.shape
    T = (Ys @ Ys.transpose([0, 2, 1])).mean(axis=0) / p
    S = (Ys.transpose([0, 2, 1]) @ Ys).mean(axis=0) / n
    return T, S

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
    B_approx_iters: int
):
    """
    Solves system of linear equations for the eigenvalues
    `B_approx_iters` is how many times to run the least
    squares computation on partial data.  If it is -1,
    then we run the computation on the whole data.  This
    is the most accurate, but increases space complexity
    from a quadratic to a cubic polynomial.
    
    An implementation of Lemma 3
    """
    
    (n, p) = Sigmas.shape
    invSigs = 1 / Sigmas
    
    a = invSigs.T.reshape((n*p,))
    if B_approx_iters == -1:
        # Most accurate, but increases space complexity
        # from n^2 + p^2 to pn^2 + np^2 !!
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
    else:
        # Less accurate, 
        Ls = np.zeros((n+p,))
        for it in range(B_approx_iters):
            a_ = np.empty((n+p,))
            B_ = np.zeros((n+p, n+p))
            for row in range(n + p):
                # First, figure out what row
                # we want from the full B
                if row < n:
                    # Get all terms involving ith eigenvector of Psi
                    true_row = it*n + row
                else:
                    # Get all terms involving ith eigenvector of Theta
                    true_row = it + (row-n)*n
                i = true_row % n
                j = true_row // n
                B_[row, :] = 0
                B_[row, i] = 1
                B_[row, n+j] = 1
                a_[row] = a[true_row]
            Ls += np.linalg.lstsq(B_, a_, rcond=None)[0]
        Ls /= B_approx_iters
    
    return Ls[:n], Ls[n:]

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
    
    
    return u, v