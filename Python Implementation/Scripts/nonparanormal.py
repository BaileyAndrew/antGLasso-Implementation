import numpy as np
from scipy.stats import norm # for norm.ppf, the inverse cdf of normal distribution

def nonparanormal(
    Ys: "(m, n, p) tensor",
) -> (
    "(m, n, n) tensor, approximately normally distributed"
):
    """
    Implements the nonparanormal trick from the paper
    "The Nonparanormal: Semiparametric Estimation of High Dimensional Undirected Graphs"
    by Liu, Lafferty, and Wasserman
    """
    
    m, n, p = Ys.shape
    
    # In the paper, they have n samples and p datapoints per sample
    # whereas we have m samples and ell=n*p datapoints per sample
    
    delta_m = 1 / (4 * np.power(m, 1/4) * np.sqrt(np.pi * np.log(m)))
    
    mu: "(n, p)" = Ys.mean(axis=0)
    sigma: "(n, p)" = Ys.std(axis=0)
    
    # Note that in the paper, F_hat_j just counts average number of elements in the
    # jth position that the element at the jth position is less than or equal to
    # Note that 4 - np.argsort([11, 10, 12, 9]) gives [2, 3, 1, 4] which is
    # exactly what we want! (4 comes from length of list)
    F_hat = 1 - np.argsort(Ys, axis=0) / m
   
    # Now truncate to get F_tilde
    F_hat[F_hat < delta_m] = delta_m
    F_hat[F_hat > 1 - delta_m] = 1 - delta_m
    
    return mu + sigma * norm.ppf(F_hat)