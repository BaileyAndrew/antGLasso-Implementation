"""
niBiGLasso = "Non-Iterative" BiGraphical Lasso
Named because it's not an iterative algorithm.

Advantages:
1) Don't need to worry about convergence
2) Don't need hyperparameters
3) In practice does very well and is very fast
4) The approximation it relies on can be checked
        so we'll know if the results are bad
5) The approximation improves as number of samples
        increases, and in fact improves so quickly
        that other factors become main source of
        errors already at #samples >= 3
6) Only a single call to Lasso per precision matrix

Disadvantages:
1) It relies on an approximation
2) There is no way to adjust the quality of this
        approximation.  If it's bad, then you'll
        have to use a different algorithm.
3) I haven't yet finished proving the approximation
        is valid ;)  Empirically it is, and I have
        a good idea of how the proof should go.
4) For 1 or 2 samples, the approximation is very bad.
"""

import numpy as np
from Scripts.utilities import scale_diagonals_to_1, LASSO

# For vindication
from Scripts.utilities import tr_p, kron_sum_diag

class niBiGLasso:
    """
    Follow the scikit-learn API
    """
    
    def __init__(self, *, silence_warnings=False):
        self.vindication: "Measure of how good the approximation is" = \
            "Not Vindicated"
        self.silence_warnings = silence_warnings
        self.already_computed = False
        
        self.Psi = None
        self.Theta = None
        
        # For vindication
        self.u = None
        self.v = None
        
    def get_empiricals(
        self,
        Ys: "(m, n, p) tensor, m slices of observed (n, p) matrix Y_k"
    ) -> (
        "(n, n) within-row empirical covariance matrix",
        "(p, p) within-row empirical covariance matrix"
    ):
        """
        Compute within-row/within-column empirical covariance matrices
        
        Note that this is a MLE of the true covariance matrices
        And that they follow a Wishart distribution with m degrees
        of freedom and some scale parameter that idk at the moment.
        """
        if len(Ys.shape) == 2:
            Ys = Ys[np.newaxis, :, :]
        
        (m, n, p) = Ys.shape
        T_psi: "(Average) empirical covariance matrix for Psi"
        T_theta: "(Average) empirical covariance matrix for Theta"
        T_psi = np.einsum("mnp, mlp -> nl", Ys, Ys) / (m*p)
        T_theta = np.einsum("mnp, mnl -> pl", Ys, Ys) / (m*n)
        
        assert T_psi.shape == (n, n)
        assert T_theta.shape == (p, p)
        
        return T_psi, T_theta
        
    def fit(
        self,
        T_psi: "(n, n) within-row empirical covariance matrix",
        T_theta: "(p, p) within-column empirical covariance matrix"
    ) -> (
        "(n, n) within-row precision matrix",
        "(p, p) within-column precision matrix"
    ):
        """
        Find the within-row precision matrices Psi and Theta,
        without Lasso.
        
        Relies on tr_p[D.inv] being approximately proportional to
        tr_p[D].inv (after a scaling operation on T)
        This seems to be true when eigenvalues near 1,
        but proof pending...
        
        [That might seem like a strange approximation but it
        is empirically very justified and I have a good idea
        of how the proof should go, some bits are difficult
        but intuitive]
        """
        
        if self.already_computed and not self.silence_warnings:
            print("Warning: Already fitted with this instance")
            
        n = T_psi.shape[0]
        p = T_theta.shape[0]
            
        # Let's scale the covariance matrices to precision matrices
        # Empirically this makes their eigenvalues close to 1
        # This is important for our approximation
        #T_psi = scale_diagonals_to_1(T_psi)
        #T_theta = scale_diagonals_to_1(T_theta)
        # Hadamard multiply by the K matrices
        K_psi = (n * np.ones(T_psi.shape) + (2*n - 2) * np.eye(T_psi.shape[0]))
        #K_psi = (2*p-1)*np.eye(T_psi.shape[0]) + p*(np.ones(T_psi.shape)-np.eye(T_psi.shape[0]))
        K_theta = (p * np.ones(T_theta.shape) + (2*p - 2) * np.eye(T_theta.shape[0]))
        T_psi_ = T_psi * K_psi
        T_theta_ = T_theta * K_theta

        # Calculate the eigendecomposition
        ell_psi, U = np.linalg.eig(T_psi_)
        ell_theta, V = np.linalg.eig(T_theta_)

        # This approximates tr_p[D].inv, which seems to be
        # approximately colinear with tr_p[D.inv] (the quantity
        # that we actually want). [colinearity is all we need]
        # But if we treat our values as tr_p[D].inv then we
        # have a non-iterative solution for the eigenvalues.
        ell_psi = 1 / ell_psi
        ell_theta = 1 / ell_theta

        # Construct the matrix that relates these to the eigenvalues
        X = np.ones((n + p, n + p))
        X[:p, :p] = n * np.eye(p)
        X[p:, p:] = p * np.eye(n)

        # Find eigenvalues
        ell = np.concatenate((ell_psi, ell_theta))
        lmbda = np.linalg.lstsq(X, ell, rcond=None)[0]
        u = lmbda[:p]
        v = lmbda[p:]
        
        # Test what if we just inverted the T evals
        # THIS DOES AMAZINGLY!
        # [Should comment it out for actual alg though]
        # note that m ~= 8 seems to be magical point where
        # it starts to do great
        u = 1 / np.linalg.eig(T_psi)[0]
        v = 1 / np.linalg.eig(T_theta)[0]
        
        # Store for later vindication
        self.u = u
        self.v = v
        
        # Reconstruct Psi, Theta
        Psi = U @ np.diag(u) @ U.T
        Theta = V @ np.diag(v) @ V.T
        self.Psi = scale_diagonals_to_1(Psi)
        self.Theta = scale_diagonals_to_1(Theta)
        
        return self.Psi, self.Theta
    
    def shrink(
        self,
        beta_1: "L1 penalty for Psi",
        beta_2: "L2 penalty for Theta",
    ) -> (
        "(n, n) within-row precision matrix",
        "(p, p) within-column precision matrix"
    ):
        """
        Apply LASSO regularization
        
        Note that we don't store the shrunk matrices,
        this class keeps track of the originals.
        """
        n = self.Psi.shape[0]
        p = self.Theta.shape[0]
        
        Psi = scale_diagonals_to_1(LASSO(np.eye(n), self.Psi, beta_1 / n))
        Theta = scale_diagonals_to_1(LASSO(np.eye(p), self.Theta, beta_2 / p))
        
        return Psi, Theta
    
    def vindicate(self) -> "Pair of number between 0 and 1":
        """
        To check (after-the-fact) whether the approximation was
        satisfied, we can check the absolute value of the
        cosine between the relevant values.
        
        The closer this is to 1, the more confident we can be.
        
        First in pair is for Psi estimate, second is for Theta.
        """
        
        # Internal function to reduce code duplication
        def _internals(u, v):
            p = v.shape[0]

            harmonics_first = np.diag(tr_p(np.diag(1 / kron_sum_diag(u, v)), p=p)) / (p*p)
            arithmetics_first = 1 / np.diag(tr_p(np.diag(kron_sum_diag(u, v)), p=p))
            norm = harmonics_first @ arithmetics_first
            return np.abs(norm / (
                np.linalg.norm(harmonics_first, ord=2)
                * np.linalg.norm(arithmetics_first, ord=2)
            ))
        
        return _internals(self.u, self.v), _internals(self.v, self.u)
    
    def print_vindication(self):
        a, b = self.vindicate()
        print(f"Psi vindication: {a}")
        print(f"Theta vindication: {b}")
     
# This function calls all the things you need to make a prediction.
# Can pass in a premade instance of niBiGLasso or not.
def no_hassle(Ys, beta_1, beta_2, vindicate=True, nibig=None):
    if nibig is None:
        nibig = niBiGLasso()
    T_psi, T_theta = nibig.get_empiricals(Ys)
    nibig.fit(T_psi, T_theta)
    Psi, Theta = nibig.shrink(beta_1, beta_2)
    if vindicate:
        vinds = nibig.vindicate()
    else:
        vinds = None
    return Psi, Theta, vinds
    