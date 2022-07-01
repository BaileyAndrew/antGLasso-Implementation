import numpy as np
from Scripts.utilities import K, LASSO

import scipy.sparse as sparse

def anBiGLasso(
    Ys: "(m, n, p) input tensor",
    beta_1: "L1 penalty for Psi",
    beta_2: "L1 penalty for Theta",
    B_approx_iters: (int, "Hyperparameter") = 10,
):
    """
    See `calculateEigenvalues` for explanation of
    `B_approx_iters`.
    """
    (m, n, p) = Ys.shape
        
    T, S = calculate_empirical_covariances(Ys)
    U, V = eigenvectors_MLE(T, S)
    u, v = eigenvalues_MLE(Ys, U, V, B_approx_iters)
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
    Psi: "Matrix to shrink",
    b: "L1 penalty",
    mode: ("Row by Row", "Upper Triangle") = "Row by Row (direct)"
) -> "L1-shrunk Psi":
    
    if mode == "Row by Row":
        n = Psi.shape[0]
        for r in range(n):
            row = np.delete(Psi[r, :], r, axis=0)
            row = LASSO(np.eye(n-1), row, b)
            Psi[r, :r] = row[:r]
            Psi[r, r+1:] = row[r:]
            Psi[:, r] = Psi[r, :]
    elif mode == "Row by Row (direct)":
        # Here we take advantage of the fact that our Lasso matrix is the
        # identity, and so we temporarily make our row have only positive
        # values so that the minima can be computed in closed form
        n = Psi.shape[0]
        for r in range(n):
            row = np.delete(Psi[r, :], r, axis=0)
            #next two lines same as:
            # row = np.sign(row) * LASSO(np.eye(n-1), np.abs(row), b)
            row_ = (np.abs(row) - b / 2)
            row_[row_ < 0] = 0
            row = np.sign(row) * row_
            Psi[r, :r] = row[:r]
            Psi[r, r+1:] = row[r:]
            Psi[:, r] = Psi[r, :]
    elif mode == "Upper Triangle":
        n = Psi.shape[0]
        s = (Psi.size - n)//2
        tridx = np.triu_indices_from(Psi, 1)
        Psi[tridx] = LASSO(np.eye(s), Psi[tridx], b * s)
        Psi = Psi + Psi.T - np.diag(np.diag(Psi))
    else:
        raise ValueError(f"No such mode `{mode}`")
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
    B_approx_iters: int,
    init_v = None
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
    
    a = invSigs.reshape(-1, order='F')
    
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
        # Slightly less accurate, but much faster and more memory efficient
        Ls = np.zeros((n+p,))
        
        if init_v is not None:
            Ls[-p:] = init_v
        
        #B = np.eye(n + p, n + p)
        #B[n:-1, -2] = 1
        #B[:n, -1] = 1
        #B[-2, -1] = 1
        
        # It may be worth creating this as a sparse array,
        # but the version of SciPy I'm using (1.7.3) does
        # not have good tools for it - 1.8 does, but pandas
        # breaks when I use 1.8 at the moment.
        #B_inv = np.eye(n + p, n + p)
        #B_inv[n-1:-2, -2] = -1
        #B_inv[:n, -1] = -1
        #B_inv[n-1:-2, -1] = 1
        #B_inv[-2, -1] = -1
 
        # Use sparse matrices
        # Note: this will need * for matmul instead of @
        B_csr = sparse.eye(n+p, n+p, format='lil')
        B_csr[n-1:-2, -2] = -1
        B_csr[:n, -1] = -1
        B_csr[n-1:-2, -1] = 1
        B_csr[-2, -1] = -1
        B_csr = sparse.csr_matrix(B_csr)
        #print(B_csr.todense())
        
        #print(B_inv)
        
        #B_approx_iters=1#DEBUG
        for it in range(B_approx_iters):
            # Select the ith eigenvalue of Psi,
            # and the jth eigenvalue of Theta
            j = np.random.randint(0, n)
            i = np.random.randint(0, p)
                
            # ignore what came before
            # We can work out that the duplicate row occurs
            # when, for integer x, j+x*n is in [i*n, i*n+n)
            # We can use this to work out the index to delete
            #print("========")
            #print(a:=np.arange(a.shape[0])) # TEST
            #j = 2 # TEST
            #i = 2 # TEST
            a_ = np.concatenate([
                a[i*n:i*n+n],
                np.delete(a[j::n], ((i*n + n - j - 1) // n))
            ])
            
            #print(a_)
            
                
            a_[j:] = np.roll(a_[j:], -1)
            
            #print(a_)
            #print('-')
            
            # Add previous estimate for eigenvalue corresponding
            # to the last column to the end of a_, so that
            # we can solve an invertible square matrix!
            a_2 = np.empty((n + p,))
            a_2[:n+p-1] = a_
            a_2[-1] = (Ls[i+n-1] / it) if it > 0 else 1
            
            #print(a_2)
            
            # The version of scipy this was written in only has good
            # support for sparse "matrices", not "arrays" - what
            # this means for the reader is that '*' corresponds to
            # '@' here.  It is matrix multiplication, not elementwise.
            out = B_csr * a_2
            
            #print(out)
            
            # Move last two cols back to i, j positions
            out[i+n-1:] = np.roll(out[i+n-1:], 1)
            out[j:] = np.roll(out[j:], 1)
            
            #print(out)
            
            Ls += out
        Ls /= B_approx_iters
    
    return Ls[:n], Ls[n:]

def eigenvalues_MLE(
    Ys: "(m, n, p) tensor",
    U: "(n, n) eigenvectors of Psi",
    V: "(p, p) eigenvectors of Theta",
    B_approx_iters: int,
    for_testing_params = None
):
    """
    An implementation of Theorem 2
    """
    Xs = (rescaleYs(Ys, U, V))
    Sigmas = calculateSigmas(Xs)
    
    u, v = calculateEigenvalues(Sigmas, B_approx_iters)
    return u, v