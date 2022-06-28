import numpy as np
from scipy.linalg import solve_triangular
from Scripts.utilities import K

def antGLasso(
    Ys: "(n, d_1, ..., d_K) input tensor",
    *,
    betas: ("L1 penalties for Psis", "Hyperparameter") = None,
    B_approx_iters: (int, "Hyperparameter") = 10,
    sparsities: ("List of numbers of edges to keep for Psis", "Hyperparameter") = None
):
    """
    See `calculateEigenvalues` for explanation of
    `B_approx_iters`.
    """
    
    if betas is None and sparsities is None:
        raise ValueError("betas and sparsities cannot both be None")
    if betas is not None and sparsities is not None:
        raise ValueError(
            "Must choose to regularize using either betas or sparsities, not both"
        )
    
    (n, *d) = Ys.shape
    K = len(d)
        
    Ss = [nmode_gram(Ys, ell) for ell in range(K)]
    Vs = eigenvectors_MLE(Ss)
    vs = eigenvalues_MLE(Ys, Vs, B_approx_iters)
    Psis = (V @ np.diag(v) @ V.T for V, v in zip(Vs, vs))
    
    if betas is not None:
        Psis = shrink(Psis, betas)
    if sparsities is not None:
        Psis = shrink_sparsities(Psis, sparsities)
    
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
    vs = calculateEigenvalues(Sigmas, B_approx_iters)
        
    return vs

def calculateEigenvalues(Sigmas, B_approx_iters):
    ds = np.array(Sigmas.shape)
    K = len(ds)
    a = (1 / Sigmas).reshape(-1, order='F')
    B_inv = create_B_inverse(ds)
        
    Ls = np.zeros(sum(ds))
    
    for it in range(B_approx_iters):
        a_vals = a.copy()
        # Select random eigenvalues
        idxs = np.random.randint(0, ds)
        
        ell_vals = np.arange(np.sum(ds))

        for i, val in enumerate(idxs):
            offset = np.sum(ds[:i])
            ell_vals[offset:val+1+offset] = np.roll(ell_vals[offset:val+1+offset], 1)

        for i, val in enumerate(idxs):
            chunk_size = np.prod(ds[:i])
            num_chunks = np.prod(ds[i+1:])

            # Break up into chunk_size blocks
            #split_mat = np.array(np.split(a_vals, num_chunks))
            split_mat = a_vals.reshape(num_chunks, -1) # TEST
            temp = split_mat[:, :chunk_size].copy()
            split_mat[:, :chunk_size] = split_mat[:, val*chunk_size:(val+1)*chunk_size]
            split_mat[:, val*chunk_size:(val+1)*chunk_size] = temp
            a_vals = split_mat.reshape(-1)

        shrunk = a_vals[0:1] # First row
        a_vals = a_vals[1:]
        for i, val in enumerate(ds):
            step_size = np.prod(ds[:i])
            amount = val-1
            shrunk = np.concatenate([
                shrunk,
                a_vals[0::step_size][:amount]
            ])
            a_vals = a_vals[step_size*amount:]

        for i, val in enumerate(idxs):
            # We subtract `i` to account for the fact that we've
            # already moved earlier columns!
            offset = np.sum(ds[:i])-i
            ell_vals[offset:] = np.roll(ell_vals[offset:], -1)

        out = B_inv @ np.concatenate([
            shrunk,
            Ls[ell_vals[-K+1:]] / it if it >= 1 else 0*Ls[ell_vals[-K+1:]]+1
        ])
        
        Ls += out
    Ls /= B_approx_iters
    return [Ls[np.sum(ds[:ell]):np.sum(ds[:ell+1])] for ell in range(K)]
    

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

def shrink_sparsities(
    Psis: "List of matrices to shrink",
    sparsities: "List assumed sparsities"
) -> "List of sparsity-shrunk Psis":
    out = []
    for Psi, s in zip(Psis, sparsities):
        Psabs = np.abs(Psi)
        np.fill_diagonal(Psabs, 0)
        quant = np.quantile(Psabs, 1-s)
        Psi[Psabs < quant] = 0
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