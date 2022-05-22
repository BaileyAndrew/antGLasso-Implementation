# Not implemented yet!
def nonparanormal_skeptic(
    Ys: "(m, n, p) tensor",
) -> (
    "(m, n, n) tensor - m batches of empirical covariances YYt",
    "(m, p, p) tensor - m batches of empirical covariances YtY"
):
    """
    Implements the nonparanormal skeptic
    which calculates the empirical covariance matrices
    without assuming Ys come from a matrix normal distribution
    """
    pass