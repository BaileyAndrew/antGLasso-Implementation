# Installation

We recommend installing `numpy` with intel's blas backend:
```
python -m pip install -i https://pypi.anaconda.org/intel/simple --extra-index-url https://pypi.org/simple numpy==1.22.2
```

This will make the code run much faster (only tested on a Mac).

```
# Other essential dependencies:
conda install -c conda-forge scipy=1.7.3
conda install scikit-learn=1.0.2
```

```
# Non-essential dependencies, for the notebooks
conda install matplotlib
conda install pandas=1.4.2
conda install line_profiler
```

If you want to compare results to EiGLasso, follow the instructions on their GitHub page for how to install the correct version of MKL and how to compile.  We only use their matlab interface, so the
instructions for compiling that are the most essential.

# Analytic Tensor Graphical Lasso (antGLasso)

This is a non-iterative algorithm for estimating the two constituent
precision matrices of matrix data (within-row and within-column precisions).

There are two variants of the algorithm - one with optimal space complexity
(whose performance is dependent on a hyperparameter `b` controlling the goodness
of the approximation, here we set `b=10`) and one with poor space complexity
but slightly better performance.  The latter one is called 'Hungry anBiGLasso'.
For brevity, this ReadMe won't cover Hungry antGLasso, but the repo does contain
precision-recall plots in the `Plots` folder.

## Other BiGraphical and Tensor-Graphical Lasso Algorithms

We have a custom implementation of scBiGLasso (one of the dependencies of scBiGLasso
is not compileable on a Mac since Matlab dropped support for all free Fortran compilers).
We also have added EiGLasso and TeraLasso a git submodules,
the main repositories are at https://github.com/SeyoungKimLab/EiGLasso and https://github.com/kgreenewald/teralasso.
To compare against EiGLasso (which is implemented in C++), we interface through Matlab
following the instructions of the authors of EiGLasso.  This could add some overhead,
but since antGLasso implemented in Python (a language not known for speed)
we are hopeful that this does not substantially affect the analysis.
We use the out-of-the-box hyperparameters for all algorithms.

## Practical Performance

### Summary

Based on the results of my experiments, I would recommend EiGLasso when speed is not an issue,
and antGLasso if speed becomes an issue.  anBiGLasso can handle a 1600x1600 dataset in
less than a minute, whereas EiGLasso can only handle a 300x300 dataset in the same timeframe.
However, EiGLasso, especially on small samples, is more accurate than anBiGLasso.

### Runtimes

scBiGLasso, EiGLasso, and TeraLasso are iterative algorithms, which means their speed can
vary substantially depending on how fast they converge.  Empirically I've noticed
that you will get quick convergence (and good precision/recall) if your
precision matrices are drawn from the Inverse Wishart Distribution with degrees
of freedom being twice the size of the matrix.  If you draw from the same distribution
but with half the degrees of freedom, you will get slow-converging, poor-performing
results.  We look at both of these cases.

![All Together](Plots/Runtimes%20Comparison/All%20Together.png)

For large sample data, due to memory usage concerns we only tested up to 500x500 inputs rather than 5000x5000,
but the trend should be clear.

![All Together Tensors](Plots/Runtimes%20Comparison/All%20Together%20Tensors.png)

For 3D tensors, antGLasso is still the fastest, but TeraLasso closes the gap somewhat.  This is because
the expense of creating the empirical covariance matrices starts to become dominant as the number
of tensor dimensions increases.  As both TeraLasso and antGLasso create these matrices, the speedup
of antGLasso starts to matter less and less.

### Results on Simulated Data (Large Sample: 100 samples of 50x50)

We just show the performance on 'easy' data, as the 'hard' data results had much more variance.  However,
the `Plots` folder does contain analagous hard-data plots. 

![Large Sample Results](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Final/PR%20100%20of%2050x50.png)

We can see that antGLasso does worse than the others, but still respectably.  It is the same story in all cases we consider.

### Results on Simulated Data (Small Sample: 10 samples of 50x50)


![Small Sample Results](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Final/PR%2010%20of%2050x50.png)

### Results on Tensor Data (100 samples of 50x50x50)

Here we only compare to TeraLasso as it is the only one that can handle tensor-variate data.

![Tensor Results](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Final/PR%20100%20of%2050x50x50.png)

### Performance as Input Size Varies

![Size Results](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Final/PR%20antGLasso%20vary%20sizes.png)

We can see that, as the size of the problem gets bigger, antGLasso's performance improves.

### Peformance as Number of Samples Varies

![Samples Results](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Final/PR%20antGLasso%20vary%20samples.png)

As expected, more samples lead to better performance.

### Effect of Regularization

For this algorithm, L1 Penalties correspond to thresholds above which we preserve edges. 

![Regularization Results](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Final/PR%20antGLasso%20sparsity-based%20regularization.png)

This test was done on data with 20% sparsity.  We can see that, when we set the threshold to preserve 20% of the edges, we get
a balance of precision and recall (~60% for both).

## Performance on Real Data

_This section is under construction and subject to change - the previous section is very stable so you can trust it_

We investigate performance on two datasets - the 'Duck Dataset', using the same experiment as described in the original BiGLasso paper,
and the 'Mouse Dataset', using the same experiment as described in the scBiGLasso paper.

### Summary

Real data often comes in a single sample.  _Under construction_

To achieve these results, we occasionally increased the `b` parameter to 1000 (instead of 10).  This did not noticeably increase
runtime.  It seems on real world data there is more variance in the outputs of the algorithm, but large `b` will average these
differences out.

### Mouse Dataset

This dataset has three cell types: S, G1, and G2M.  We would expect that the precision matrix has blocks around the diagonal
indicating each of these cell types.

This data follows a count distribution, not a normal distribution, so we take the the log of our data (plus 1 to remove zeros) to
roughly map it to a normal.

#### antGLasso

![Mouse antGLasso](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Final/antGLasso%20Mouse%20Performance%20Direct%20Fit.png)

antGLasso may have learned some information about this problem (it identifies S and G2M block diagonals), but not much
- it seems to put S and G2M cells together, and does not recognize the G1 cluster.
This is unfortunate but not unexpected given the performance of the algorithm on small sample data.


#### EiGLasso

We can then compare antGLasso's performance against EiGLasso - if they give similar results, this serves as partial validation of antGLasso.

![Mouse EiGLasso](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Final/EiGLasso%20Mouse%20Performance%202.png)

This does similarly to antGLasso, although the clusters are more refined (there's less 'bleed' into what should be the G1 cluster).  With different regularization penalties, we can achieve this:

![Mouse EiGLasso](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Final/EiGLasso%20Mouse%20Performance%201.png)

Where it seems to have identified the block-diagonal structure except for the S group strongly interacting with the others.

#### antGLasso Animation

antGLasso separates the estimation and regularization steps, unlike other algorithms.  This can be a disadvantage, as it will
not benefit from task transfer.  However, the silver lining is that it allows us to easily see how our results change with different
penalties, as the regularization step is very cheap.

![Mouse antGLasso](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Final/antGLasso%20Mouse.gif)

### COIL Dataset

We would expect each frame to be conditionally dependent on nearby frames, manifesting in a precision matrix that hugs the diagonal.
We might also expect the upper right and lower left corners to be nonzero, as the video is of a duck rotating around 360°.  We also
might expect there to be a repeating structure every 64 pixels in the pixel dependencies, as that is the size of a row of pixels.

The results align well with our expectations.  However, we could not use the traditional antGLasso algorithm on it.  Unlike the
mouse embryo case, the data follows a zero-inflated binomial distribution.  (Well, it's more of a 'pixel-values-less-than-ten-are-inflated'
binomial distribution), due to the background of the image being dark and nondescript.  This is harder to map to a Gaussian (one
reason is that it has two peaks).  We weren't able to get good results using a handmade transformation, so we instead used a heuristic
for the eigenvalues of the precision matrix.  This heuristic antGLasso can accept empirical covariance matrices as input, allowing
us to apply the nonparanormal skeptic.

In the performance graphs from the previous section, the heuristic is called `anBiGLasso_cov` _(I'll change name later)_.  Its precision
recall curves are comparable to EiGLasso's, except that it has an upper bound on the recall, that gets lower as the problem size
increases.  This is because of the way the heuristic works, unfortunately.  So it's not an ideal solution.

![Duck](https://github.com/BaileyAndrew/scBiGLasso-Implementation/blob/main/Plots/Final/antGLasso%20Duck%20Heuristic.png)

Note: in the original BiGraphical Lasso paper, they shrunk the 128x128 pixel frames to 9x9.  We use 64x64, which takes about 20
seconds (as our input data is of size (70, 4096)).  Smaller cases, such as 32x32, take neglibigle runtime.  We wanted to consider
the full 128x128 case, but the memory requirements would be prohibitive on my machine (it would require 2 gigabytes of input,
2 gigabytes of output, and then some intermediate matrices also of size 2 gigabytes each that could possibly be optimized away with
some effort).

Given a machine
with sufficient memory, extrapolating the cubic time complexity implies it would take $20 \times 4^3$ seconds, or 21 minutes to run.
This is a very reasonable runtime, indicating that the limits of this algorithm are bounded by memory not speed.  Since the space
complexity of anBiGLasso is optimal, this is a fundamental limit of the BiGraphical Lasso problem itself rather than our algorithm.


## Data

Given precision matrices, we generate gaussian matrix data using the matrix-variate normal with
Kronecker Sum structure.  The precision matrices are generated using a 'base' positive definite
matrix variate distribution (specifically the Inverse Wishart), and then 'sparsifying' them
by Hadamard-producting them with some positive definite masking matrix represented as the
outer product of an i.i.d. vector of bernoulli variables with itself (and then manually setting
the diagonals to 1).

For real data, the relevant notebooks in this repo contain download links.

