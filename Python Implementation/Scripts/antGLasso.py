import numpy as np
from Scripts.utilities import K

import scipy.sparse as sparse


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
    
    (n, *ds) = Ys.shape
    K = len(ds)
        
    Ss = [nmode_gram(Ys, ell) for ell in range(K)]
    Vs = eigenvectors_MLE(Ss)
    vs = eigenvalues_MLE(Ys, Vs, B_approx_iters)
    Psis = (V @ np.diag(v) @ V.T for V, v in zip(Vs, vs))
    
    if betas is not None:
        Psis = shrink(Psis, betas)
    if sparsities is not None:
        Psis = shrink_sparsities(Psis, sparsities)
    
    return Psis

def antGLasso_heuristic(
    Ss: "List of empirical covariance matrices",
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
        
    Vs = eigenvectors_MLE(Ss)
    vs = eigenvalues_heuristic(Ss, Vs, B_approx_iters)
    Psis = [V @ np.diag(v) @ V.T for V, v in zip(Vs, vs)]
    
    if betas is not None:
        Psis = shrink(Psis, betas)
    if sparsities is not None:
        Psis = shrink_sparsities(Psis, sparsities)
    
    return Psis

def eigenvalues_heuristic(Ss, Vs, B_approx_iters):
    ds = [S.shape[0] for S in Ss]
    vs = [np.zeros(d) for d in ds]
    for idx, S in enumerate(Ss):
        V = Vs[idx]
        d = ds[idx]
        ds_slash_d = [d for i, d in enumerate(ds) if i != idx]
        Sigmas = np.prod(ds_slash_d) * np.diag(V.T @ S @ V)
        Sigmas = np.tile(
            Sigmas.reshape(d, *[1 for _ in ds_slash_d]),
            (1, *ds_slash_d)
        )
        
        vs[idx], *_ = calculateEigenvalues(Sigmas, B_approx_iters)
 
    return vs

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
    tdims = np.sum(ds)
    a_vals = (1 / Sigmas).reshape(-1, order='F')

    B_csr = sparse.eye(tdims, tdims, format='lil')
    B_csr[-K, -K+1:] = -1
    B_csr[0:ds[0]-1, -K+1:] = -1
    B_csr[ds[0]-1:-K, -K] = -1
    r = ds[0]-1
    for idx, d in enumerate(ds):
        if idx==0:
            continue
        next_r = r+d-1
        B_csr[r:next_r, -K+idx] = 1
        r = next_r
    B_csr = sparse.csr_matrix(B_csr)
        
    Ls = np.zeros(tdims)
    
    
    for it in range(B_approx_iters):
        # Select random eigenvalues
        idxs = np.random.randint(0, ds)
        
        ell_vals = np.arange(np.sum(ds))

        # Move columns so that guesses are the first column
        for i, val in enumerate(idxs):
            offset = np.sum(ds[:i])
            ell_vals[offset:val+1+offset] = np.roll(ell_vals[offset:val+1+offset], 1)


        rows = []
        chunk_size = np.array([np.prod(ds[:ell]) for ell in range(K)])
        csidx = chunk_size @ idxs
        for ell, d in enumerate(ds):
            start = csidx - idxs[ell]*np.prod(ds[:ell])
            rows += [idx for k in range(d) if (idx:=start + k*chunk_size[ell]) != csidx]
        rows += [csidx]
        shrunk = a_vals[rows]
        
        for i, val in enumerate(idxs):
            # We subtract `i` to account for the fact that we've
            # already moved earlier columns!
            offset = np.sum(ds[:i])-i
            ell_vals[offset:] = np.roll(ell_vals[offset:], -1)
            
        # B_inv @ to_mult.....
        # but `B_csr` is a "matrix type" i.e. uses * for matmul
        to_mult = np.concatenate([
            shrunk,
            Ls[ell_vals[-K+1:]] / it if it >= 1 else 0*Ls[ell_vals[-K+1:]]+1
        ])
        
        out = np.zeros(to_mult.shape)
        out[ell_vals] = B_csr * to_mult
        
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
        np.moveaxis(A, n+1, 1),
        (A.shape[0], A.shape[n+1], -1), # The -1 infers the value (m_n)
        order='F' # Do math vectorization order rather than numpy vectorization order
    )
    return (An @ An.transpose([0, 2, 1])).mean(axis=0) / An.shape[-1]