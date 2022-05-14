# scBiGLasso-Implementation
A Python implementation of the Scalable Bigraphical Lasso algorithm.  Implementations/wrappers for other languages, such as R, may be added later.

Note that I do not currently implement the nonparanormal sceptic layer,
although I will add that at some point so that I can test this on actual count data!

# Performance

My computer:
* MacBook Pro (13-inch, M1, 2020)
* 8 GB RAM


## Matrix-Variate Data Generation

Generating 100 samples of (100, 100) matrix variate data from a KS-structured Matrix Normal takes ~0.75 seconds.
The number of samples does affect this - about 30% of the runtime is due to there being 100 samples.

_This has been well-optimized but still has avenues for improvement.  I did try batching, i.e. generate (5, 100, 100, 100) samples
where the first index has different precision matrices, the second index is how many samples from same distribution.  This would
have helped in exploring how penalties, sizes, and densities affect precision/recall.  Unfortunately, I run out of memory for
even the smallest batches :(_

About 70% of the runtime is in a kronecker product (custom implementation that outperforms `np.kron`), and 30% is due to a
matrix multiplication.  If low number of samples, nearly all of the runtime is from the kronecker product.

## Matrix Decomposition

Given 100 samples of (100, 100) matrix variate data, calculating Psi/Theta takes 1.5-2 seconds.  The number of samples is
effectively irrelevant for this.

Currently, half the runtime takes place inside `scikit-learn`'s Lasso implementation, and I'm unlikely to be able to do anything
about that.  I did try `cvxpy`'s implementation, which was notably slower.  The other half takes place in a single call to
`np.einsum` in the calculation of the A\i\i matrix.  It's possible that this could be improved, but I think it would require
a deeper understanding of the costs of matrix operations than I currently posses.

# Observations

I'm still making some performance improvements so I have not focused too much on actually using it.  However, I have noticed that,
for a precision matrix density of 10%, the algorithm gets better the larger the precision matrix.
