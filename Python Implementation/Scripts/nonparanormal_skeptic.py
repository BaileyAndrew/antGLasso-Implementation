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