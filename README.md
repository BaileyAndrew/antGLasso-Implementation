# Analytic BiGraphical Lasso

This is a non-iterative algorithm for estimating the two constituent
precision matrices of matrix data (within-row and within-column precisions).

There are two variants of the algorithm - one with optimal space complexity
(whose performance is dependent on a hyperparameter `b` controlling the goodness
of the approximation, here we set `b=10`) and one with poor space complexity
but better performance.  The latter one is called 'Hungry anBiGLasso'.

Also implemented are the scBiGLasso algorithm and _(not yet but soon)_ the
TeraLasso algorithm, for comparisons.

## Practical Performance

_(TODO: Compare with TeraLasso)_

All comparisons are done using a large number of samples.  For a small number of samples,
results may be different.

### Runtimes

scBiGLasso and TeraLasso are iterative algorithms, which means their speed can
vary substantially depending on how fast they converge.  Empirically I've noticed
that you will get quick convergence (and good precision/recall) if your
precision matrices are drawn from the Inverse Wishart Distribution with degrees
of freedom being twice the size of the matrix.  If you draw from the same distribution
but with half the degrees of freedom, you will get slow-converging, poor-performing
results.  We look at both of these cases.

#### 'Easy' Data (Quick Convergence)

![EasyData](Plots/Runtimes%20Comparison/Compare%20Runtimes%20Easy%20Data.png)

#### 'Hard' Data (Slow Convergence)

![HardData](Plots/Runtimes%20Comparison/Compare%20Runtimes%20Hard%20Data.png)

### Results

_(TODO: Look at performance on real data)_

You get roughly the same precision-recall curves regardless of the size of the input
data (but the best L1 penalties will be different).  We can see that anBiGLasso gets
roughly the same results as scBiGLasso - the hungry algorithm does slightly better,
whereas the well-nourished (?) algorithm does slightly worse.

#### anBiGLasso Results

![anBiGLasso Results](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Vary%20Sizes%20-%20anBiGLasso%20-%20Easy%20-%20Approx/Precision-Recall-Vary-Sizes-100.png)

##### Hungry anBiGLasso

![hungry anBiGLasso Results](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Vary%20Sizes%20-%20anBiGLasso%20-%20Easy/Precision-Recall-Vary-Sizes-100.png)

#### scBiGLasso Results

![scBiGLasso Results](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Vary%20Sizes%20-%20scBiGLasso%20-%20Easy/Precision-Recall-Vary-Sizes-100.png)

However, these results are just for the aforementioned 'easy' data.  For hard data, the
results are substantially worse:

#### Hard anBiGLasso Results

_todo: nonhungry hard results_

##### Hungry Hard anBiGLasso Results

![anBiGLasso Results](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Vary%20Sizes%20-%20anBiGLasso%20-%20Hard/Precision-Recall-Vary-Sizes-40.png)

#### Hard scBiGLasso Results

_(It takes 5 hours to generate this graph!  B/c it's slow on hard data)_

![scBiGLasso Results](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Vary%20Sizes%20-%20scBiGLasso%20-%20Hard/Precision-Recall-Vary-Sizes-40.png)

We can see that the scBiGLasso results on 'hard' data are nonsense - this is because the algorithm
did not converge within 100 iterations, so we cut it off.

## Asymptotic Performance

The hungry implementation of anBiGLasso has the unideal space complexity $O(mnp + n^2p + p^2n)$,
but the non-hungry version has a space complexity of $O(mnp + n^2 + p^2)$
(the size of the inputs plus the outputs).

Time complexity is $O(mn^2p + mnp^2 + b(n^3 + p^3 + n^2p + p^2n))$.  In practice $b$ is constant,
it does not seem to need to be increased for larger data.  (However, it could be that the best
value grows with the log of the data, I haven't investigated...)

## Data

Given precision matrices, we generate gaussian matrix data using the matrix-variate normal with
Kronecker Sum structure.  The precision matrices are generated using a 'base' positive definite
matrix variate distribution (specifically the Inverse Wishart), and then 'sparsifying' them
by Hadamard-producting them with some positive definite masking matrix represented as the
outer product of an i.i.d. vector of bernoulli variables with itself (and then manually setting
the diagonals to 1).

The Nonparanormal Skeptic layer is not currently implemented, in fact it is nontrivial to implement
as our algorithm needs the raw data to work - it cannot be framed as a function of covariance matrices.

