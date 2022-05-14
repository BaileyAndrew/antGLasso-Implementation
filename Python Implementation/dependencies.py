"""
This script just prints the versions of all required dependencies
"""

import numpy as np # For matrix maths
import cvxpy as cp # For lasso regression
import scipy # For probability distributions
import sklearn # For lasso regression

import sys # For getting python version [Standard library]


print(f"Python Version {sys.version}")
print(f"Numpy Version {np.__version__}")
print(f"Cvxpy Version {cp.__version__}")
print(f"Scipy Version {scipy.__version__}")
print(f"Sklearn Version {sklearn.__version__}")