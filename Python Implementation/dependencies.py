"""
This script just prints the versions of all dependencies
"""

import numpy as np # For matrix maths
import cvxpy as cp # For lasso regression
import scipy # For probability distributions
import sklearn # For lasso regression
import matplotlib # For graphs of experiments
import line_profiler # For optimizing runtime
import memory_profiler # For optimizing memory usage


import sys # For getting python version [Standard library]


print(f"Python Version {sys.version}")
print(f"Numpy Version {np.__version__}")
print(f"Cvxpy Version {cp.__version__}")
print(f"Scipy Version {scipy.__version__}")
print(f"Sklearn Version {sklearn.__version__}")
print(f"Matplotlib Version {matplotlib.__version__}")
print(f"Line Profiler Version {line_profiler.__version__}")
print(f"Memory Profiler Version {memory_profiler.__version__}")