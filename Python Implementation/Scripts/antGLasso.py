import numpy as np
from scipy.linalg import solve_triangular
from Scripts.utilities import K, LASSO

def antGLasso(
    Ys: "(n, d_1, ..., d_K) input tensor",
    betas: "L1 penalties for Psis",
    B_approx_iters: (int, "Hyperparameter") = 10,
):
    """
    See `calculateEigenvalues` for explanation of
    `B_approx_iters`.
    """
    (n, *d) = Ys.shape
    K = len(d)
        
    Ss = [nmode_gram(Ys, ell) for ell in range(K)]
    Vs = eigenvectors_MLE(Ss)
    vs = eigenvalues_MLE(Ys, Vs, B_approx_iters)
    Psis = shrink(
        (V @ np.diag(v) @ V.T for V, v in zip(Vs, vs)),
        betas
    )
    
    return Psis

def eigenvectors_MLE(Ss):
    mk = 1
    for S in Ss:
        mk *= S.shape[0]
    Vs = []
    for S in Ss:
        p = S.shape[0]
        n = mk / p
        _S = S * K(p, 2*n-1, n)
        _, V = np.linalg.eigh(_S)
        Vs.append(V)
    return Vs

def eigenvalues_MLE(Ys, Vs, B_approx_iters):
    Xs = rescaleYs(Ys, Vs)
    Sigmas = calculateSigmas(Xs)
    
    u, v = calculateEigenvalues(Sigmas, B_approx_iters)
    return u, v

def calculateEigenvalues(Sigmas, B_approx_iters):
    d = np.array(Sigmas.shape)
    K = len(d)
    a = (1 / Sigmas).reshape(-1, order='F')
    B_inv = create_B_inverse(d)
        
    Ls = np.zeros(sum(d))
    
    B_approx_iters=1#DEBUG
    for it in range(B_approx_iters):
        # Select random eigenvalues
        #idxs = np.random.randint(0, d)
        idxs = [0, 0]

        # ignore what came before
        # We can work out that the duplicate row occurs
        # when, for integer x, j+x*n is in [i*n, i*n+n)
        # We can use this to work out the index to delete
        print('---')
        print(a)
        
        """
        # Get first row
        first_row = np.sum(idxs[ell]*np.prod(d[:ell]) for ell in range(K))
        a_ = a[first_row:first_row+1]
        a = np.delete(a, first_row, axis=0)
        """
        
        a_ = a[0:1]
        a = a[1:]
        step = 1
        for ell in range(K):
            print(a_)
            a_ = np.concatenate([
                a_,
                a[idxs[ell]::step][:d[ell]-1]
            ])
            a = a[step:]
            step *= d[ell]
        
        """
        for ell in range(K):
            offset = np.sum([idxs[n]*np.sum(d[:n]) for n in range(K) if n != ell])
            step = np.prod(d[ell])
            #print(offset)
            #print(step)
            a_ = np.concatenate([
                a_,
                np.delete(a[offset::step][:d[ell]], idxs[ell])
            ])
        """
        
        if it == 0:
            # If no previous guesses, initialize to 1
            a_ = np.concatenate([
                a_,
                np.ones(K-1)
            ])
        else:
            for ell in range(K-1):
                # Add previous guesses
                a_ = np.concatenate([
                    a_,
                    np.array([Ls[np.sum(d[:ell]) + idxs[ell]] / (B_approx_iters - 1)])
                ])
                
        print(a_)
                
        # Apply P_2 matrix
        a_[:np.sum(d)-K+1] = np.roll(a_[:np.sum(d)-K+1], -1)
        
        # Apply B^-1 matrix
        out = B_inv @ a_
        
        
        # Apply P_1 matrix
        """
        idxs_ = [idxs[ell] + np.sum(d[:ell]) for ell in range(K)]
        move_cols = out[idxs] # creates copy of columns to put on end
        print(np.delete(out, idxs_))
        out = np.concatenate([
            np.delete(out, idxs_),
            move_cols
        ])
        print(out)
        """
        idxs_ = [idxs[ell] + np.sum(d[:ell]) for ell in range(K)]
        for ell in range(K):
            out[idxs_[-(ell+1)]-(K-ell-1):] = np.roll(
                out[idxs_[-(ell+1)]-(K-ell-1):],
                1
            )
        Ls += out
        print(out)
    Ls /= B_approx_iters
    return [Ls[np.sum(d[:ell]):np.sum(d[:ell+1])] for ell in range(K)]
    

def rescaleYs(
    Ys: "(n, d_1, ..., d_K) tensor",
    Vs: "List of (d_ell, d_ell) eigenvectors of Psi_ell",
) -> "(n, d_1, ..., d_K) tensor":
    """
    Rescales our input data to be drawn from a kronecker sum
    distribution with parameters being the eigenvalues
    
    An implementation of Lemma 1
    """
    n, *d = Ys.shape
    K = len(d)
    Xs = Ys
    for k in range(0, K):
        # Shuffle important axis to end, multiply, then move axis back
        Xs = np.moveaxis(
            np.moveaxis(
                Xs,
                k+1,
                -1
            ) @ Vs[k],
            -1,
            k+1
        )
    return Xs

def calculateSigmas(
    Xs: "(n, d_1, ..., d_k) tensor"
) -> "(n, d_1, ..., d_k) tensor OF variances":
    """
    Gets an MLE for variances of our rescaled Ys
    
    An implementation of Lemma 2
    """
    
    (n, *d) = Xs.shape
    return ((Xs**2).sum(axis=0) / n)

def shrink(
    Psis: "List of matrices to shrink",
    bs: "List of L1 penalties"
) -> "List of L1-shrunk Psis":
    out = []
    for Psi, b in zip(Psis, bs):
        n = Psi.shape[0]
        for r in range(n):
            row = np.delete(Psi[r, :], r, axis=0)
            row_ = (np.abs(row) - b / 2)
            row_[row_ < 0] = 0
            row = np.sign(row) * row_
            Psi[r, :r] = row[:r]
            Psi[r, r+1:] = row[r:]
            Psi[:, r] = Psi[r, :]
        out.append(Psi)
    return out

def nmode_gram(A, n):
    An = np.reshape(
        np.moveaxis(A, n, 0),
        (A.shape[n], -1), # The -1 infers the value (m_n)
        order='F' # Do math vectorization order rather than numpy vectorization order
    )
    return An @ An.T

def create_B_inverse(shp):
    K = len(shp)
    I_block_shp = np.sum(shp) - K
    I_block = np.eye(I_block_shp)
    Null_block = np.zeros((K, I_block_shp))
    Lower_I_block = np.eye(K, K)
    Lower_I_block[0, 1:] = -1
  
    Rest = np.ones((I_block_shp, K))
    zero_idx = 0
    for idx, s in enumerate(shp):
        update = s-1
        Rest[zero_idx:zero_idx+update, idx] = 0
        zero_idx += update
        
        
    return np.concatenate([
        np.concatenate([
            I_block,
            Null_block
        ], axis=0),
        np.concatenate([
            -Rest @ Lower_I_block,
            Lower_I_block
        ], axis=0)
    ], axis=1)

def nmode_gram(A, n):
    An = np.reshape(
        np.moveaxis(A, n+1, 1),
        (A.shape[0], A.shape[n+1], -1), # The -1 infers the value (m_n)
        order='F' # Do math vectorization order rather than numpy vectorization order
    )
    return (An @ An.transpose([0, 2, 1])).mean(axis=0) / An.shape[-1]