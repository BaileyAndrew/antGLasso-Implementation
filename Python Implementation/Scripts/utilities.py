"""
Contains helper functions
"""

import numpy as np
import cvxpy as cp
from sklearn import linear_model

def tr_p(A: "Matrix", p: "Contraction size"):
    """
    Calculates the blockwise trace, i.e. break the matrix
    up into p by p blocks and form the matrix of traces of
    those blocks
    """
    (r, c) = A.shape
    assert r % p == 0 and c % p == 0, \
        f"Dimensions mismatch: {r, c} not contractible by {p}"
    out = np.empty((r // p, c // p))
    for x in range(r // p):
        for y in range(c // p):
            out[x, y] = np.trace(A[x*p:(x+1)*p, y*p:(y+1)*p])
    return out
    
def kron_sum(A, B):
    """
    Computes the kronecker sum of two square input matrices
    
    Note: `scipy.sparse.kronsum` is a thing that would
    be useful - but it seems that `scipy.sparse` is not
    yet a mature library to use.
    """
    a, _ = A.shape
    b, _ = B.shape
    return np.kron(A, np.eye(b)) + np.kron(np.eye(a), B)

def kron_sum_diag(a, b):
    """
    Computes the diagonal of the kronecker sum of two diagonal matrixes,
    given as input the diagonals of the two input matrices
    
    Always returns a column vector, i.e. shape of form (x, 1)
    """
    # Remove nuisance dimensions
    a = a.squeeze()
    b = b.squeeze()
    
    m = a.shape[0]
    n = b.shape[0]
    
    # Result is sum of [a1 ... a1 a2 ... a2 ... am ... am]
    # and [b1 b2 ... bn b1 b2 ... bn ... b1 b2 ... bn]
    # To get the second vector, we can tile a.
    # To get the first vector, we can tile b in a second dimension and
    # then flatten it in a way contrary to the tiling.
    A = np.tile(a, (n, 1)).T.reshape((1, n*m))
    B = np.tile(b, (1, m))
    return A + B

def LASSO_cvxpy(
    X: "Coefficient matrix",
    y: "Affine vector",
    lmbda: "L1 penalty",
    **kwargs: "Does nothing"
):
    """
    Lasso regression
    Much of this code taken from CVXPY tutorials
    """
    _, n = X.shape
    beta = cp.Variable(n)
    
    def loss_fn(X, y, beta):
        return cp.norm2(X @ beta - y)**2

    def regularizer(beta):
        return cp.norm1(beta)

    def objective_fn(X, Y, beta, lambd):
        return loss_fn(X, y, beta) + lambd * regularizer(beta)

    def mse(X, Y, beta):
        return (1.0 / X.shape[0]) * loss_fn(X, y, beta).value
    
    problem = cp.Problem(cp.Minimize(objective_fn(X, y, beta, lmbda)))
    problem.solve(solver='ECOS', warm_start=True)
    return beta.value
    
def LASSO_sklearn(
    X: "Coefficient matrix",
    y: "Affine vector",
    lmbda: "L1 penalty",
    **kwargs: "To pass to sklearn"
):
    """
    Lasso regression using scipy
    """
    # pre-creating these with warm_start for each row does not
    # improve speed unfortunately.
    lasso = linear_model.Lasso(alpha=lmbda, fit_intercept=False, **kwargs)
    try:
        return lasso.fit(X, y).coef_
    except ValueError as e:
        print(X)
        raise e

def LASSO(
    X: "Coefficient matrix",
    y: "Affine vector",
    lmbda: "L1 penalty",
    **kwargs: "To pass to underlying package"
):
    """
    Choose which method of Lasso to use
    """
    return LASSO_sklearn(X, y, lmbda, **kwargs)


def precision(
    cm: "[[TP, FP], [FN, TN]]"
):
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

def recall(
    cm: "[[TP, FP], [FN, TN]]"
):
    return cm[0, 0] / (cm[0, 0] + cm[1, 0])

def accuracy(
    cm: "[[TP, FP], [FN, TN]]"
):
    return np.trace(cm) / cm.sum()

def F1_score(
    cm: "[[TP, FP], [FN, TN]]"
):
    prec = precision(cm)
    rec = recall(cm)
    return 2 * prec * rec / (prec + rec)

def binarize_matrix(
    M: "Input matrix of any dimensions",
    eps: "Tolerance" = 0.001,
    mode: "Negative | <Tolerance" = '<Tolerance'
):
    """
    Returns M but with only ones and zeros
    """
    out = np.empty(M.shape)
    if mode == '<Tolerance' or mode == 'Nonzero':
        out[np.abs(M) <= eps] = 0
        out[np.abs(M) > eps] = 1
    elif mode == 'Negative':
        out[M < 0] = 1
        out[M >= 0] = 0
        
        # In negative mode we need to add diagonals back
        out += np.eye(out.shape[0])
    else:
        raise Exception(f'Invalid mode {mode}')
    return out

def generate_confusion_matrices(
    pred: "Square matrix",
    truth: "Square matrix",
    eps: "Tolerance" = 0.001,
    mode: "See docstring" = '<Tolerance'
) -> "(2, 2) confusion matrix [[TP, FP], [FN, TN]]":
    """
    `mode`:
        '<Tolerance': x->1 if |x| < eps, else x->0
        'Nonzero': Same as '<Tolerance'
        'Negative': x->1 if x < 0 else x -> 0
        'Mixed': truth is '<Tolerance', pred is 'Negative'
    """
    mode_pred = mode if mode != 'Mixed' else 'Negative'
    mode_truth = mode if mode != 'Mixed' else 'Nonzero'
    pred = binarize_matrix(pred, eps=eps, mode=mode_pred)
    truth = binarize_matrix(truth, eps=0, mode=mode_truth)
    
    # Identity matrices to remove diagonals
    In = np.eye(pred.shape[0])
    
    TP: "True positives"
    TP = (pred * truth - In).sum()
    
    FP: "False positives"
    FP = (pred * (1 - truth)).sum()
    
    TN: "True negatives"
    TN = ((1 - pred) * (1 - truth)).sum()
    
    FN: "False negatives"
    FN = ((1 - pred) *  truth).sum()
    
    return np.array([
        [TP, FP],
        [FN, TN]
    ])