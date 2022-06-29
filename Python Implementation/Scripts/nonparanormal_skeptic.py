import numpy as np
from scipy.stats import spearmanr

def nonparanormal_skeptic(
    Ys: "(m, n, p) tensor"
):
    m, n, p = Ys.shape
    T = np.empty((m, n, n))
    S = np.empty((m, p, p))
    
    for batch in range(m):
        T[batch] = 2 * np.sin(np.pi / 6 * spearmanr(Ys[batch], axis=1)[0])
        S[batch] = 2 * np.sin(np.pi / 6 * spearmanr(Ys[batch], axis=0)[0])
        
    T = T.mean(axis=0)
    S = S.mean(axis=0)
    np.fill_diagonal(T, 1)
    np.fill_diagonal(S, 1)
    return T, S

def nonparanormal_tensor_skeptic(
    Ys: "(m, *ds)"
):
    m, *ds = Ys.shape
    Ss = [np.empty((m, d, d)) for d in ds]
    
    for idx, d in enumerate(ds):
        dim = idx+1
        Ys_dim = np.moveaxis(Ys, dim, 1)
        for batch in range(m):
            Ss[idx][batch] = 2 * np.sin(np.pi / 6 * spearmanr(
                Ys_dim[batch].reshape(-1, np.prod(ds)//d), axis=1
            )[0])
        Ss[idx] = Ss[idx].mean(axis=0)
        np.fill_diagonal(Ss[idx], 1)
    return Ss[::-1] #b/c this func made them in reversed order