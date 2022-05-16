# scBiGLasso-Implementation
A Python implementation of the Scalable Bigraphical Lasso algorithm.  Implementations/wrappers for other languages, such as R, may be added later.

Note that I do not currently implement the nonparanormal sceptic layer,
although I will add that at some point so that I can test this on actual count data!

# Observations

I'm still making some performance improvements so I have not focused too much on actually using it.  However, I have noticed that,
for a precision matrix density of 10%, the algorithm gets better the larger the precision matrix.

Also, if you have a small number of samples, it seems the algorithm takes much longer to converge.

# Performance

My computer:
* MacBook Pro (13-inch, M1, 2020)
* 8 GB RAM


## Matrix-Variate Data Generation

Generating 100 samples of (100, 100) matrix variate data from a KS-structured Matrix Normal takes ~0.75-1 seconds.
The number of samples does affect this - about 30% of the runtime is due to there being 100 samples.

_This has been well-optimized but still has avenues for improvement.  I did try batching, i.e. generate (5, 100, 100, 100) samples
where the first index has different precision matrices, the second index is how many samples from same distribution.  This would
have helped in exploring how penalties, sizes, and densities affect precision/recall.  Unfortunately, I run out of memory for
even the smallest batches :(_

About 70% of the runtime is in a kronecker product (custom implementation that outperforms `np.kron`), and 30% is due to a
matrix multiplication.  If low number of samples, nearly all of the runtime is from the kronecker product.

## Matrix Decomposition

Given 100 samples of (100, 100) matrix variate data, calculating Psi/Theta takes ~0.9-1 seconds.  The number of samples is
responsible for about 10% of this (from calculating the mean sample).  It used to be that a small number of samples
would take much longer, because it would take more iterations to converge.  However, I managed to reformulate the problem
such that there is only ever one Lasso call per matrix (rather than one call per matrix per row per iteration), i.e. 2
calls overall.  The large sample case remains the same timewise, but the small sample case now only takes ~1 second as well,
because the iterations are much faster.

Currently, ~40% of the runtime comes from calculating eigenvalues/eigenvectors, ~20% from calculating the A matrix, and
~40% from `np.linalg.lstsq`-ing an equation to find the pseudoinverse of A.  Perhaps I could improve the calculation of
the A matrix (although I've already done a lot there), but the rest are unlikely to be improvable b/c they are fairly
fundamental operations.  Any future improvements would likely need to come from improving the math of the problem rather than
the implementation of the math.

I had to assume that the diagonals of Psi/Theta were 1 to achieve this speedup.  Since diagonals can't be determined
by the algorithm, that's not a big deal.  It however does prevent us from re-computing the eigenvalues/vectors every
loop in the update of Psi/Theta.  This is not a big deal, large-sample code runs faster if we don't (I never tested
small-samples though, and it could plausibly affect it then since small-samples seem to take many iterations).
