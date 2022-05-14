# scBiGLasso-Implementation
A Python implementation of the Scalable Bigraphical Lasso algorithm.  Implementations/wrappers for other languages, such as R, may be added later.

Note that I do not currently implement the nonparanormal sceptic layer,
although I will add that at some point so that I can test this on actual count data!

# Performance

My computer:
* MacBook Pro (13-inch, M1, 2020)
* 8 GB RAM


## Matrix-Variate Data Generation

Generating 100 samples of (100, 100) matrix variate data from a KS-structured Matrix Normal takes 2-2.5 seconds.  The number of samples is
effectively irrelevant for this.

_This has been well-optimized but still has avenues for improvement.
Especially since most of my tests are actually on data of the shape (attempts, samples, n, p) where attempts is the number of precision matrix
pairs to generate, samples is the number of samples to generate for each (Psi, Theta) pair, and then n, p are sizes of each sample.  There is
the opportunity to make this code run quickly over batches._

Currently, about 90% of the runtime takes place in a call to `np.kron`.  Conveniently, the next version of Numpy (1.23) contains
improvements to this function so that it will be about 5x as fast on the test suite they used.  Although when I move to batch
computation I won't be able to use this, and will rather use an `np.einsum` based solution - so the point may be moot.

## Matrix Decomposition

Given 100 samples of (100, 100) matrix variate data, calculating Psi/Theta takes 1.5-2 seconds.  The number of samples is
effectively irrelevant for this.

_This has been aggressively optimized and it's hard to see where I could improve it.  There is opportunity to make this run
on batches as well, just like for data generation, but because some in the batch may converge sooner than others it could
actually be a non-negligible headache._

Currently, half the runtime takes place inside `scikit-learn`'s Lasso implementation, and I'm unlikely to be able to do anything
about that.  I did try `cvxpy`'s implementation, which was notably slower.  The other half takes place in a single call to
`np.einsum` in the calculation of the A\i\i matrix.  It's possible that this could be improved, but I think it would require
a deeper understanding of the costs of matrix operations than I currently posses.

# Observations

I'm still making some performance improvements so I have not focused too much on actually using it.  However, I have noticed that,
for a precision matrix density of 10%, the algorithm gets better the larger the precision matrix.
