"""
This script calculates scBiGLasso
"""

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from Scripts.utilities import LASSO
import warnings

# Note: in matrix variable name subscripts:
# 'sisj' will represent '\i\j'
# i.e. s = \


def _calculate_A(
    i: "Row of precision matrix we're currently estimating",
    U: "Eigenvectors of Psi",
    u: "Vector of eigenvalues of Psi",
    v: "Vector of eigenvalues of Theta",
    psi_ii: "Diagonal element of Psi",
    path: "The path for the einsum operation" = 'optimal'
):
    """
    The indices here are different than those used in paper.
    (Because the paper uses i,j here and elsewhere it uses i for
    other things, we chose to stay consistent with the 'other things'
    rather than this calculation)
    
    paper -> code:
    
    i -> k   [inner sum index]
    j -> ell [outer sum index]
      -> t   [row of column of A; not indexed in paper]
      -> i   [row of precision matrix; not indexed in paper]
    k -> j   [column of A]
    """
    
    n = u.shape[0]
    p = v.shape[1]
    
    # So far the best way, where we just let numpy figure out the most
    # efficient way to do the sum
    U_si = np.delete(U, i, axis=0)
    B = (1 / (psi_ii + v)).squeeze()
    C = 1 / (u + v)
    newer_way = np.einsum("l, kl, ak, bk -> ab", B, C, U_si, U_si, optimize=path)
    
    return newer_way

def _calculate_full_A(
    U: "Eigenvectors of Psi",
    u: "Vector of eigenvalues of Psi",
    v: "Vector of eigenvalues of Theta",
    path: "The path for the einsum operation" = 'optimal'
):
    """
    The indices here are different than those used in paper.
    (Because the paper uses i,j here and elsewhere it uses i for
    other things, we chose to stay consistent with the 'other things'
    rather than this calculation)
    
    paper -> code:
    
    i -> k   [inner sum index]
    j -> ell [outer sum index]
      -> t   [row of column of A; not indexed in paper]
      -> i   [row of precision matrix; not indexed in paper]
    k -> j   [column of A]
    """
    
    n = u.shape[0]
    p = v.shape[1]
    
    # So far the best way, where we just let numpy figure out the most
    # efficient way to do the sum
    B = (1 / (1 + v)).squeeze()
    C = 1 / (u + v)
    newer_way = np.einsum("l, kl, ak, bk -> ab", B, C, U, U, optimize=path)
    
    return newer_way
    
# Note: for some reason, LASSO_sklearn can fail to converge even though
# LASSO_cvxpy will converge - but LASSO_sklearn's results are better in
# that case?
warnings.filterwarnings("ignore", category=ConvergenceWarning)
def _scBiGLasso_internal(
    Psi: "Previous estimate of precision matrix",
    Theta: "Other precision matrix [taken as constant]",
    T: "Estimated covariance matrix",
    beta: "L1 penalty",
    path: "Contraction order for A's einsum calculation" = 'optimal',
    recalculate_eigs: "Do we recompute eigvals/vecs every loop" = False,
    verbose: bool = False
):
    out_Psi = Psi.copy()
    n, _ = Psi.shape
    p, _ = Theta.shape
    
    U: "Eigenvectors of Psi"
    u: "Diagonal eigenvalues of Psi"
    u, U = np.linalg.eigh(out_Psi)
    u = u.reshape((n, 1))
    
    V: "Eigenvectors of Theta"
    v: "Diagonal eigenvalues of Theta"
    v, V = np.linalg.eigh(Theta)
    v = v.reshape((1, p))
    
    A = _calculate_full_A(U, u, v)
    
    for i in range(0, n):
        # Loop through rows of Psi
        
        if recalculate_eigs and i != 0:
            U: "Eigenvectors of Psi"
            u: "Diagonal eigenvalues of Psi"
            u, U = np.linalg.eigh(out_Psi)
            u = u.reshape((n, 1))
        
        psi_ii = out_Psi[i, i]

        # Estimate new psi_isi
        A_sisi: "A_\i\i as in paper" = np.delete(np.delete(A, i, axis=0), i, axis=1)
        t_isi = np.delete(T, i, axis=1)[i, :]
        
        # Note that the paper has A @ psi + p * t = 0
        # But sklearn minimizes A @ psi - p * t
        # Hence the factor of -p we apply.
        psi_isi_update = LASSO(A_sisi, -p * t_isi, beta)
        
        # Update row
        out_Psi[i, :i] = psi_isi_update[:i]
        out_Psi[i, (i+1):] = psi_isi_update[i:]
        
        # It's symmetric, so update column too
        out_Psi[:, i] = out_Psi[i, :]
        
    if verbose:
        u, U = np.linalg.eigh(out_Psi)
        u = u.reshape((n, 1))
        log_det: "log|kronsum(Psi, Theta)|" = np.log(u + v).sum()
    else:
        log_det: "dummy value, not used" = 0
    return out_Psi, log_det

def scBiGLasso(
    N: "Maximum iteration number",
    eps: "Tolerance",
    Ys: "m by p by n tensor, m slices of observed p by n matrix Y_k",
    beta_1: "Psi's L1 penalty",
    beta_2: "Theta's L1 penalty",
    Psi_init: "n by n initial estimate for Psi" = None,
    Theta_init: "p by p initial estimate for Theta" = None,
    verbose: bool = False
) -> ("Psi", "Theta"):
    # If m=1 (only one observation), we allow p by n matrix as input
    # by just adding an extra initial dimension of length 1
    if len(Ys.shape) == 2:
        Ys = Ys[np.newaxis, :, :]
        
    (m, p, n) = Ys.shape
    T_psi: "(Average) empirical covariance matrix for Psi"
    T_theta: "(Average) empirical covariance matrix for Theta"
    T_psi = np.einsum("mpn, mpl -> nl", Ys, Ys) / (m*p)
    T_theta = np.einsum("mpn, mln -> pl", Ys, Ys) / (m*n)
    
    if Psi_init is None:
        Psi_init = T_psi
    if Theta_init is None:
        Theta_init = T_theta
        
    # Now we make sure that the diagonals are 1, since it allows
    # a simplification later on in the algorithm
    D_psi = np.diag(1 / np.sqrt(np.diag(Psi_init)))
    Psi_init = D_psi @ Psi_init @ D_psi
    D_theta = np.diag(1 / np.sqrt(np.diag(Theta_init)))
    Theta_init = D_theta @ Theta_init @ D_theta
    
    Psi = Psi_init
    Theta = Theta_init
    
    # Used to speed up computation of the A matrix prior to Lasso
    path_Psi: "Contraction order for the A matrix" = np.einsum_path(
        "l, kl, ak, bk -> ab",
        np.empty((p,)),
        np.empty((n, p)),
        np.empty((n-1, n)),
        np.empty((n-1, n)),
        optimize='optimal'
    )[0]
    path_Theta: "Contraction order for the A matrix" = np.einsum_path(
        "l, kl, ak, bk -> ab",
        np.empty((n,)),
        np.empty((p, n)),
        np.empty((p-1, p)),
        np.empty((p-1, p)),
        optimize='optimal'
    )[0]
    
    # Will maintain a list to check for early convergence
    old_convergence_checks = [eps, eps]
    
    for tau in range(N):
        old_Psi = Psi.copy()
        old_Theta = Theta.copy()
        
        # Estimate Psi
        Psi, _ = _scBiGLasso_internal(Psi, Theta, T_psi, beta_1, path_Psi, verbose)
        
        # Now estimate Theta
        Theta, log_det = _scBiGLasso_internal(Theta, Psi, T_theta, beta_2, path_Theta, verbose)
            
        # Keep track of objective function, if verbose
        if verbose:
            psi_obj = p * np.trace(Psi @ T_psi) - 0.5 * log_det
            psi_lasso = psi_obj + beta_1 * np.abs(Psi).sum()
            theta_obj = n * np.trace(Theta @ T_theta) - 0.5 * log_det
            theta_lasso = theta_obj + beta_2 * np.abs(Theta).sum()
            obj = psi_obj + theta_obj
            full_obj = psi_lasso + theta_lasso
            print(f"Iter={tau}: NLL={obj:.5f}, w/ Lasso: {full_obj:.5f}")
            print(f"Just Psi: NLL={psi_obj:.5f}, w/ Lasso: {psi_lasso:.5f}")
            print(f"Just Theta: NLL={theta_obj:.5f}, w/ Lasso: {theta_lasso:.5f}")
            print('-------')
            
        # Check convergence
        if tau >= 2:
            old_convergence_checks.append(
                max(
                    np.linalg.norm(Psi - old_Psi, ord='fro')**2,
                    np.linalg.norm(Theta - old_Theta, ord='fro')**2
                )
            )
            if max(old_convergence_checks) < eps:
                if verbose:
                    print(f"Early convergence on iteration {tau}!")
                break
            old_convergence_checks = old_convergence_checks[1:]
    
    return Psi, Theta
