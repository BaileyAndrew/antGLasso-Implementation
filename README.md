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

I did some mild math tricks to do a whole-matrix lasso instead of a row-by-row lasso, which was a major speedup.  This results
in a single lasso call per flip-flop.  I then removed the lasso per flip-flop, having only a single lasso at the very end.
This means that we're finding the dense solution's fixed point, and then lassoing it (as opposed to approaching the fixed point
with the lasso solution).  An effect of this is that it seems there is more variance in the quality of the outputs.
It typically runs in a fraction of a second (0.3-0.6), but if you're unlucky it may run in a little
over a second (1.3) if it takes many iterations to solve.

**Possible improvements?**: This has been very heavily optimized - I'm even directly calling LAPACK (Fortran) functions in some places!
There are two `np.einsum` calls that I might try to express in terms of LAPACK operations (since the matrices are often symmetric,
LAPACK can be an order of magnitude faster for some computations by leveraging this).  Other than that, all the runtime takes place
in the calculation of the eigenvalues/vectors.  I implemented this in LAPACK but had no gain (so I reverted to scipy).  If LAPACK
can't make it faster, then there's no way I can ;)  So other than the `np.einsum`s, the only real improvements could be improvements
to the algorithm itself rather than the implementation of the algorithm.  That's a bit beyond my capabilities.
