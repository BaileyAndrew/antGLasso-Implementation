"""
This script contains functions to generate matrix-variate data
"""

import numpy as np
from scipy.stats import wishart, matrix_normal, bernoulli
from scipy.stats import multivariate_normal
from scipy.linalg import solve_triangular
from Scripts.utilities import kron_sum, kron_sum_diag

def multi_norm(
    precision: "n by n Precision Matrix",
    size: "Amount of samples"
) -> "`size` samples of zero-mean multivariate normal with `precision` matrix":
    
    
    # My solution follows second answer from:
    # https://stackoverflow.com/questions/16706226/
    #    how-do-i-draw-samples-from-multivariate-gaussian-distribution-parameterized-by-p
    
    # Basically, if LL^T = Q,
    # then Lz = u where u ~ multivariate with precision Q
    # given that z ~ multivariate independent
    
    n = precision.shape[0]
    
    # Note that this has shape (n, size) because this way
    # L.T@z = v is batch-multiplying over the columns of z
    # (even though typically (size, n) would be standard format)
    z: "`size` independent samples of n-dimensional i.i.d. gaussian vector"
    z = multivariate_normal(cov=1).rvs(
        size=size*n
    ).reshape(n, size)
    
    L: "Lower triangular Cholesky decomposition of precision matrix"
    L = np.linalg.cholesky(precision)
    
    # We have to transpose the output because v is (n, size)
    # instead of (size, n).
    cholesky_solution = solve_triangular(
        L.T,
        z,
        lower=False,
        check_finite=False,
        overwrite_b=True
    ).T
    
    return cholesky_solution

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
    # Equivalent to Lam_inv = np.sqrt(1 / np.diag(kron_sum(np.diag(u), np.diag(v))))
    # but ~3x as fast
    Lam_inv = np.sqrt(1 / kron_sum_diag(u, v))
    
    A: "Matrix to map i.i.d. gaussian to Omega-precision gaussian"
    # 90% of computational cost comes from np.kron in this expression
    A = np.kron(U, V) * Lam_inv
    
    # Note that shape is (n, size) instead of standard (size, n)
    # to make the batched matrix multiplication easier
    z: "`size` independent samples of n-dimensional i.i.d. gaussian vector"
    z = multivariate_normal(cov=1).rvs(
        size=size*(n*p)
    ).reshape(n*p, size)
    
    # Transpose to return to standard shape (size, n)
    Ys_vec: "Vectorized version of output matrix" = (A @ z).T
    Ys = np.transpose(Ys_vec.reshape((size, n, p)), [0, 2, 1])
    
    return Ys

def generate_sparse_posdef_matrix(
    n: "Number of rows/columns of output",
    expected_nonzero: "Number of nondiagonal nonzero entries expected",
    *,
    off_diagonal_scale: "Value strictly between 0 and 1 to guarantee posdefness" = 0.9,
    size: "Number of samples to return" = 1
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

    Psi = wishart.rvs(100, np.eye(n), size=size) / 100 * Mask
    Psi /= np.trace(Psi, axis1=1, axis2=2).reshape(size, 1, 1) / n
    
    return Psi

def generate_Ys(
    m: "Number of Samples",
    p: "Number of Datapoints",
    n: "Number of Features",
    expected_nonzero_psi: "Number of nondiagonal nonzero entries expected in Psi",
    expected_nonzero_theta: "Number of nondiagonal nonzero entries expected in Theta",
    off_diagonal_scale: "Value strictly between 0 and 1 to guarantee inverse" = 0.9,
    structure: "Kronecker Sum/Product" = "Kronecker Sum"
) -> "(n, n) precision matrix, (p, p) precision matrix, (m, p, n) sample tensor":
    
    """
    Generates two sparse positive definite matrices.
    Relies on Schur Product Theorem; we create a positive definite mask matrix and
    then hadamard it with our precision matrices
    
    Then generate m samples of p by n matrices from the matrix normal (kronecker
    sum or kronecker product) distribution using these two precision matrices
    """
    
    p_psi: "Bernoulli probability to achieve desired expected value of psi nonzeros"
    p_psi = np.sqrt(expected_nonzero_psi / (n**2 - n))
    p_theta: "Bernoulli probability to achieve desired expected value of theta nonzeros"
    p_theta = np.sqrt(expected_nonzero_theta / (n**2 - n))
    
    Psi_mask: "Mask to zero out elements while preserving pos. definiteness"
    Psi_b = bernoulli(p=p_psi).rvs(size=(n, 1)) * np.sqrt(off_diagonal_scale)
    Psi_D = np.diagflat(1 - Psi_b * Psi_b)
    Psi_mask = Psi_D + Psi_b @ Psi_b.T
    
    Theta_mask: "Mask to zero out elements while preserving pos. definiteness"
    Theta_b = bernoulli(p=p_theta).rvs(size=(p, 1)) * np.sqrt(off_diagonal_scale)
    Theta_D = np.diagflat(1 - Theta_b * Theta_b)
    Theta_mask = Theta_D + Theta_b @ Theta_b.T

    Psi_gen = wishart.rvs(100, np.eye(n)) / 100 * Psi_mask
    Theta_gen = wishart.rvs(100, np.eye(p)) / 100 * Theta_mask
    Psi_gen /= np.diag(Psi_gen).mean()
    Theta_gen /= np.diag(Theta_gen).mean()
    
    if structure == "Kronecker Product":
        Ys = matrix_normal(
            rowcov=np.linalg.inv(Theta_gen),
            colcov=np.linalg.inv(Psi_gen)
        ).rvs(size=m)
    elif structure == "Cholesky Kronecker Sum":
        # This method has been superseded by one based on SVD
        # Will leave this here for a bit in case I can think of a way
        # to optimize it, since their runtimes are comparable
        # (SVD method is about 6x as fast currently)
        Ys_vec = multi_norm(kron_sum(Psi_gen, Theta_gen), m)
        Ys = np.transpose(Ys_vec.reshape((m, n, p)), [0, 2, 1])
    elif structure == "Kronecker Sum":
        # Based on SVD.  In Numpy 1.23 they made `np.kron` 5x faster,
        # which will should improve the runtime of this method when
        # Numpy 1.23 is released as 90% of its runtime is currently
        # taken up by a single call to `np.kron`.
        Ys = matrix_normal_ks(Psi_gen, Theta_gen, m)
    elif structure == "Inefficient Kronecker Sum":
        # Generates from first principles, very slow for medium-large inputs
        # but useful when testing speedups to the `Kronecker Sum` structure
        # as we can use this as a ground truth to compare against.
        Omega_gen: "Combined precision matrix" = kron_sum(Psi_gen, Theta_gen)
        Sigma_gen: "Covariance matrix" = np.linalg.inv(Omega_gen)
        Ys_vec = multivariate_normal(cov=Sigma_gen).rvs(size=m)
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
        
    return Psi_gen, Theta_gen, Ys
    