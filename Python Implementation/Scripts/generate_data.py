"""
This script contains functions to generate matrix-variate data
"""

import numpy as np
from scipy.stats import wishart, invwishart, matrix_normal, bernoulli
from scipy.stats import multivariate_normal
from scipy.linalg import solve_triangular
from Scripts.utilities import kron_sum, kron_sum_diag, kron_prod

def matrix_normal_ks(
    Psi: "(Positive Definite) (n, n) Precision Matrix",
    Theta: "(Positive Definite) (p, p) Precision Matrix",
    size: "Number of Samples"
):
    """
    Kronecker Sum structured matrix-variate gaussian distribution
    """
    
    n = Psi.shape[0]
    p = Theta.shape[0]
    
    u, U = np.linalg.eigh(Psi)
    v, V = np.linalg.eigh(Theta)    
    
    Omega: "kronsum(Psi, Theta) - need not be calculated"
    Lam_inv: "Square root of matrix of eigenvalues of inverse of Omega" 
    Lam_inv = np.sqrt(1 / kron_sum_diag(u, v))
    
    A: "Matrix to map i.i.d. gaussian to Omega-precision gaussian"
    A = kron_prod(U, V) * Lam_inv
    
    # Note that shape is (n, size) instead of standard (size, n)
    # to make the batched matrix multiplication easier
    z: "`size` independent samples of np-dimensional i.i.d. gaussian vector"
    z = multivariate_normal(cov=1).rvs(
        size=size*(n*p)
    ).reshape(n*p, size)
    
    # Transpose to return to standard shape (size, n)
    Ys_vec: "Vectorized version of output matrix" = (A @ z).T
    #Ys = np.transpose(Ys_vec.reshape((size, n, p)), [0, 2, 1])
    Ys = Ys_vec.reshape((size, n, p))
    
    return Ys

def fast_matrix_normal_ks(
    Psi: "(Positive Definite) (n, n) Precision Matrix",
    Theta: "(Positive Definite) (p, p) Precision Matrix",
    size: "Number of Samples"
):
    """
    Kronecker Sum structured matrix-variate gaussian distribution
    
    Based on Lemma 1
    """
    
    n = Psi.shape[0]
    p = Theta.shape[0]
    
    u, U = np.linalg.eigh(Psi)
    v, V = np.linalg.eigh(Theta)    
    
    diag_precisions = kron_sum_diag(u, v)
    
    # Note that shape is (n, size) instead of standard (size, n)
    # to make the batched matrix multiplication easier
    z: "`size` independent samples of np-dimensional gaussian vector"
    z = multivariate_normal(cov=1).rvs(
        size=size*(n*p)
    ).reshape(size, n*p) / np.sqrt(diag_precisions)
    
    Xs: "Sample of diagonalized distribution" = z.reshape(size, n, p)
    Ys: "Sample of true distribution" = U @ Xs @ V.T
    
    return Ys

def fast_tensor_normal_ks(
    Psis: "List of (d_i, d_i) precision matrices, of length K >= 2",
    size: "Number of samples"
) -> "Kronecker sum distributed tensor":
    K = len(Psis)
    ds = [Psi.shape[0] for Psi in Psis]
    vs, Vs = zip(*[np.linalg.eigh(Psi) for Psi in Psis])
    
    diag_precisions = vs[0]
    for idx, v in enumerate(vs):
        if idx == 0:
            # Already accounted for vs[0]
            continue
        diag_precisions = kron_sum_diag(diag_precisions, v)
    
    z = multivariate_normal(cov=1).rvs(
        size=size*np.prod(ds)
    ).reshape(size, np.prod(ds)) / np.sqrt(diag_precisions)
    
    Xs: "Sample of diagonalized distribution" = z.reshape(size, *ds)
    
    for k in range(K):
        Xs = np.moveaxis(
            np.moveaxis(Xs, k+1, -1) @ Vs[k].T,
            -1,
            k+1
        )
    return Xs

def generate_sparse_posdef_matrix(
    n: "Number of rows/columns of output",
    expected_nonzero: "Number of nondiagonal nonzero entries expected",
    *,
    off_diagonal_scale: "Value strictly between 0 and 1 to guarantee posdefness" = 0.9,
    size: "Number of samples to return" = 1,
    df_scale: "How much to multiply the df parameter of invwishart, must be >= 1" = 1
) -> "(`size`, n, n) batch of sparse positive definite matrices":
    """
    Generates two sparse positive definite matrices.
    Relies on Schur Product Theorem; we create a positive definite mask matrix and
    then hadamard it with our precision matrices
    """
    
    Psi: "Sparse posdef matrix - the output"
    
    p: "Bernoulli probability to achieve desired expected value of psi nonzeros"
    p = np.sqrt(expected_nonzero / (n**2 - n))
    
    # Note that in the calculation of D, we make use of Numpy broadcasting.
    # It's the same as hadamard-producting with np.eye(n) tiled `size` times
    # in the size dimension and 1-b*b `n` times in the -1 dimension,
    # which is equivalent to making a batch of diagonal matrices from
    # entries of b.
    Mask: "Mask to zero out elements while preserving pos. definiteness"
    b = bernoulli(p=p).rvs(size=(size, n, 1)) * np.sqrt(off_diagonal_scale)
    D = (1-b*b)*np.eye(n)
    Mask = D + b @ b.transpose([0, 2, 1])
    Psi = invwishart.rvs(df_scale * n, np.eye(n), size=size) / (df_scale * n) * Mask
    #Psi = wishart.rvs(df_scale * n, np.eye(n), size=size) / (df_scale * n) * Mask
    
    # This just affects how much normalization is needed
    Psi /= np.trace(Psi, axis1=1, axis2=2).reshape(size, 1, 1) / n
    
    return Psi
    
def generate_Ys(
    m: "Number of Samples",
    ds: "List of shapes of precision matrices",
    *,
    expected_nonzero: "Number of nondiagonal nonzero entries expected in Psis",
    off_diagonal_scale: "Value strictly between 0 and 1 to guarantee inverse" = 0.9,
    df_scale: "How much to multiply the df parameter of invwishart, must be >= 1" = 1
) -> "List of precision matrices, (m, *ds) sample tensor":
    Psis = []
    for d in ds:
        Psi: "(d, d)" = generate_sparse_posdef_matrix(
            d,
            expected_nonzero,
            off_diagonal_scale=off_diagonal_scale,
            size=1,
            df_scale=df_scale
        ).squeeze()
        Psis.append(Psi)
        
    Ys = fast_tensor_normal_ks(Psis, m)
    if (m > 1):
        Ys -= Ys.mean(axis=0)
        
    return Psis, Ys

def generate_Ys_old(
    m: "Number of Samples",
    p: "Number of Datapoints",
    n: "Number of Features",
    expected_nonzero_psi: "Number of nondiagonal nonzero entries expected in Psi",
    expected_nonzero_theta: "Number of nondiagonal nonzero entries expected in Theta",
    off_diagonal_scale: "Value strictly between 0 and 1 to guarantee inverse" = 0.9,
    structure: "Kronecker Sum/Product" = "Kronecker Sum",
    df_scale: "How much to multiply the df parameter of invwishart, must be >= 1" = 1
) -> "(n, n) precision matrix, (p, p) precision matrix, (m, p, n) sample tensor":
    
    """
    Generate m samples of p by n matrices from the matrix normal (kronecker
    sum or kronecker product) distribution using these two precision matrices
    """
    
    Psi: "(n, n)" = generate_sparse_posdef_matrix(
        n,
        expected_nonzero_psi, 
        off_diagonal_scale=off_diagonal_scale,
        size=1,
        df_scale=df_scale
    ).squeeze()
    Theta: "(p, p)" = generate_sparse_posdef_matrix(
        p,
        expected_nonzero_theta, 
        off_diagonal_scale=off_diagonal_scale,
        size=1,
        df_scale=df_scale
    ).squeeze()
    
    if structure == "Kronecker Product":
        Ys = matrix_normal(
            rowcov=np.linalg.inv(Theta),
            colcov=np.linalg.inv(Psi)
        ).rvs(size=m)
    elif structure == "Kronecker Sum":
        # Based on SVD.
        #Ys = matrix_normal_ks(Psi, Theta, m)
        # Based on Lemma 1
        Ys = fast_matrix_normal_ks(Psi, Theta, m)
    elif structure == "Inefficient Kronecker Sum":
        # Generates from first principles, very slow for medium-large inputs
        # but useful when testing speedups to the `Kronecker Sum` structure
        # as we can use this as a ground truth to compare against.
        Omega: "Combined precision matrix" = kron_sum(Psi, Theta)
        Sigma: "Covariance matrix" = np.linalg.inv(Omega)
        Ys_vec = multivariate_normal(cov=Sigma).rvs(size=m)
        Ys = np.transpose(Ys_vec.reshape((m, n, p)), [0, 2, 1])
        # ^^ DO NOT do `Ys = Ys_vec.reshape((m, p, n))`...
        # Because numpy thinks of matrices as 'row first' whereas the
        # vectorization operation in mathematics is 'column first'
        # So the reshaping will not work like you think!
        # I spent a week debugging this T_T
    else:
        raise Exception(f"Unknown structure '{structure}'")
    
    # Remove the empirical mean
    if (m > 1):
        Ys -= Ys.mean(axis=0)
        
    return Psi, Theta, Ys
    