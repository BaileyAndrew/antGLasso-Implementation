"""
This script calculates scBiGLasso
"""

import numpy as np
import cvxpy as cp
from sklearn.exceptions import ConvergenceWarning
from Scripts.utilities import LASSO, scale_diagonals_to_1, crush_rows, uncrush_rows
from scipy.linalg import eigh
import scipy.linalg.lapack as lapack
import warnings

# Note: in matrix variable name subscripts:
# 'sisj' will represent '\i\j'
# i.e. s = \

def _calculate_A(
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
    # efficient way to do the sum.
    # Note: the 1 in 1+v is psi_ii in paper, but we assume its
    # 1 here b/c it allows us to gain a major speedup.
    B = (1 / (1 + v)).squeeze()
    C = 1 / (u + v)
    A =  np.einsum("l, kl, ak, bk -> ab", B, C, U, U, optimize=path)
    
    # A is theoretically symmetric, floating point may prevent this
    # so let's force it - because some lapack routines are faster
    # if it's symmetric
    return (A + A.T)/2
    
# Note: for some reason, LASSO_sklearn can fail to converge even though
# LASSO_cvxpy will converge - but LASSO_sklearn's results are better in
# that case?
warnings.filterwarnings("ignore", category=ConvergenceWarning)
def _scBiGLasso_internal(
    Psi: "Previous estimate of precision matrix",
    Theta: "Other precision matrix [taken as constant]",
    T_nodiag: "Estimated covariance matrix with diagonals set to zero",
    beta: "L1 penalty",
    path: "Contraction order for A's einsum calculation" = 'optimal',
    lasso_every_loop: bool = False,
    verbose: bool = False
):
    n, _ = Psi.shape
    p, _ = Theta.shape
    
    U: "Eigenvectors of Psi"
    u: "Diagonal eigenvalues of Psi"
    u, U = np.linalg.eigh(Psi) # numpy faster than scipy here
    u = u.reshape((n, 1))
    
    V: "Eigenvectors of Theta, not computed"
    v: "Diagonal eigenvalues of Theta"
    v = eigh(Theta, eigvals_only=True)
    v = v.reshape((1, p))
    A: "Used for the A_\i\i in paper" = _calculate_A(U, u, v)
    
    # Directly rely on lapack bindings to take advantage of A's symmetry!
    # This is the same as np.linalg.lstsq but *much* faster
    _, _, Psi, _ = lapack.dsysv(A, A - p * T_nodiag)
    
    if lasso_every_loop:
        # This doesn't work for some reason?  It seems it won't find
        # the correct solution if beta ~ 0 either...
        #Psi = scale_diagonals_to_1(Psi)
        #Psi_crushed = crush_rows(Psi)
        #Psi_crushed = LASSO(np.eye(n), Psi_crushed, beta / n)
        #Psi = uncrush_rows(Psi_crushed)
        
        # So we have to add to diagonal to ensure no diagonals are zero
        Psi = LASSO(np.eye(n), Psi, beta / n)
        Psi += 0.001 * np.eye(n)
        
    Psi = scale_diagonals_to_1(Psi)
        
    if verbose:
        u, U = np.linalg.eigh(Psi)
        u = u.reshape((n, 1))
        log_det: "log|kronsum(Psi, Theta)|" = np.log(u + v).sum()
    else:
        log_det: "dummy value, not used" = 0
    return (Psi + Psi.T) / 2, log_det

def scBiGLasso(
    N: "Maximum iteration number",
    eps: "Tolerance",
    Ys: "m by p by n tensor, m slices of observed p by n matrix Y_k",
    beta_1: "Psi's L1 penalty",
    beta_2: "Theta's L1 penalty",
    Psi_init: "n by n initial estimate for Psi" = None,
    Theta_init: "p by p initial estimate for Theta" = None,
    lasso_every_loop: bool = True,
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
        Psi_init = T_psi.copy()
    if Theta_init is None:
        Theta_init = T_theta.copy()
        
    # Code only depends on T without the diagonals, and
    # we can express nicer matrix equations if we set them
    # to zero, leading to a speed boost!
    T_psi -= np.diag(np.diag(T_psi))
    T_theta -= np.diag(np.diag(T_theta))
        
    # Now we make sure that the diagonals are 1, since it allows
    # a simplification later on in the algorithm
    Psi = scale_diagonals_to_1(Psi_init)
    Theta = scale_diagonals_to_1(Theta_init)
    
    # Used to speed up computation of the A matrix prior to Lasso
    path_Psi: "Contraction order for the A matrix" = np.einsum_path(
        "l, kl, ak, bk -> ab",
        np.empty((p,)),
        np.empty((n, p)),
        np.empty((n, n)),
        np.empty((n, n)),
        optimize='optimal'
    )[0]
    path_Theta: "Contraction order for the A matrix" = np.einsum_path(
        "l, kl, ak, bk -> ab",
        np.empty((n,)),
        np.empty((p, n)),
        np.empty((p, p)),
        np.empty((p, p)),
        optimize='optimal'
    )[0]
    
    # Will maintain a list to check for early convergence
    old_convergence_checks = [eps, eps]
    
    for tau in range(N):
        old_Psi = Psi.copy()
        old_Theta = Theta.copy()
        
        # Estimate Psi
        Psi, _ = _scBiGLasso_internal(
            Psi,
            Theta,
            T_psi,
            beta_1,
            path_Psi,
            lasso_every_loop,
            verbose
        )
        
        # Now estimate Theta
        Theta, log_det = _scBiGLasso_internal(
            Theta,
            Psi,
            T_theta,
            beta_2,
            path_Theta,
            lasso_every_loop,
            verbose
        )
            
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
     
    if not lasso_every_loop:
        Psi = scale_diagonals_to_1(LASSO(np.eye(n), Psi, beta_1 / n))
        Theta = scale_diagonals_to_1(LASSO(np.eye(p), Theta, beta_2 / p))
    
    return Psi, Theta


def analyticBiGLasso(
    Ys: "m by p by n tensor, m slices of observed p by n matrix Y_k",
    beta_1: "L1 penalty for Psi" = 0,
    beta_2: "L2 penalty for Theta" = 0
) -> ("Psi", "Theta"):
    """
    I think I've found an analytic solution.
    """
    if len(Ys.shape) == 2:
        Ys = Ys[np.newaxis, :, :]
        
    (m, p, n) = Ys.shape
    T_psi: "(Average) empirical covariance matrix for Psi"
    T_theta: "(Average) empirical covariance matrix for Theta"
    T_psi = np.einsum("mpn, mpl -> nl", Ys, Ys) / (m*p)
    T_theta = np.einsum("mpn, mln -> pl", Ys, Ys) / (m*n)
    
    # Hadamard multiply by the K matrices
    # Weirdly enough, these matrices work too??
    #T_psi *= (2*p - 1) * np.ones(T_psi.shape) + p * np.eye(T_psi.shape[0])
    #T_theta *= (2*n - 1) * np.ones(T_theta.shape) + n * np.eye(T_theta.shape[0])
    
    # These are the correct matrices - but it seems we don't need them???
    # Just need to boost betas if I remove them
    # For large p this is just equivalent to multiplying the diagonals by 2
    # so that could have something to do with why this step doesn't seem to
    # matter even though the maths says it does.
    T_psi *= p * np.ones(T_psi.shape) + (2*p - 2) * np.eye(T_psi.shape[0])
    T_theta *= n * np.ones(T_theta.shape) + (2*n - 2) * np.eye(T_theta.shape[0])
    
    # Calculate the eigendecomposition
    ell_psi, U = np.linalg.eigh(T_psi)
    ell_theta, V = np.linalg.eigh(T_theta)
    
    # We're really interested in the reciprocals
    ell_psi = 1 / ell_psi
    ell_theta = 1 / ell_theta
    
    # Construct the matrix that relates these to the eigenvalues
    X = np.ones((n + p, n + p))
    X[:n, :n] = p * np.eye(n)
    X[n:, n:] = n * np.eye(p)
    
    # Find eigenvalues
    ell = np.concatenate((ell_psi, ell_theta))
    # lapack.dsysv is same as np.linalg.lstsq, but faster as
    # it takes advantage of X being symmetric
    _, _, lmbda, _ = lapack.dsysv(X, ell)
    u = lmbda[:n]
    v = lmbda[n:]
    
    # Reconstruct Psi, Theta
    Psi = U @ np.diag(u) @ U.T
    Theta = V @ np.diag(v) @ V.T
    Psi = scale_diagonals_to_1(Psi)
    Theta = scale_diagonals_to_1(Theta)
        
    if beta_1 > 0: 
        Psi = scale_diagonals_to_1(LASSO(np.eye(n), Psi, beta_1 / n))
    if beta_2 > 0:
        Theta = scale_diagonals_to_1(LASSO(np.eye(p), Theta, beta_2 / p))
    
    return Psi, Theta
    
    
