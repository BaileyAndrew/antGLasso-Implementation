# Analytic BiGraphical Lasso

In `Python Implementation/Scripts/Analytic BiGLasso Minimal Example.ipynb` you can see a minimal example.  The file where the algorithm is defined is `Python Implementation/Scripts/scBiGLasso.py` - it's the last function in that file, `analyticBiGLasso`.

The rest of this readme pertains to scBiGLasso, I'll need to clean it up later but don't have time at the moment.

# scBiGLasso-Implementation
A Python implementation of the Scalable Bigraphical Lasso algorithm.  Implementations/wrappers for other languages, such as R, may be added later.

Note that I do not currently implement the nonparanormal sceptic layer,
although I will add that at some point so that I can test this on actual count data!

Note 2: I implemented the "Analytic Bigraphical Lasso" algorithm late at night so I haven't updated the readme
to reflect its existence other than this message!  You can find it in the folder `Python Implementation/Scripts/scBiGLasso.py` -
it's the last function in that file, `analyticBiGLasso`.

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

About 70% of the runtime is in a kronecker product (custom implementation that outperforms `np.kron`), and 30% is due to a
matrix multiplication.  If low number of samples, nearly all of the runtime is from the kronecker product.

## Matrix Decomposition

Runtime in the 100-sample-(100, 100)-Precision-Matrices case typically takes less than a second (can be as low as 0.3 seconds)
- if you're unlucky, it might take a bit more than a second (if it takes a long time to converge).

**Possible improvements?**: This has been very heavily optimized - I'm even directly calling LAPACK (Fortran) functions in some places!
There are two `np.einsum` calls that I might try to express in terms of LAPACK operations (since the matrices are often symmetric,
LAPACK can be an order of magnitude faster for some computations by leveraging this).  Other than that, all the runtime takes place
in the calculation of the eigenvalues/vectors.  I implemented this in LAPACK but had no gain (so I reverted to scipy).  If LAPACK
can't make it faster, then there's no way I can ;)  So other than the `np.einsum`s, the only real improvements could be improvements
to the algorithm itself rather than the implementation of the algorithm.  That's a bit beyond my capabilities.
