"""
This script prints versions of all dependencies that are only
required for the notebooks.
"""

import matplotlib # For graphs of experiments
import pandas # For loading mouse embry data
import line_profiler # For optimizing runtime
import memory_profiler # For optimizing memory usage

print(f"Matplotlib Version {matplotlib.__version__}")
print(f"Pandas Version {pandas.__version__}")
print(f"Line Profiler Version {line_profiler.__version__}")
print(f"Memory Profiler Version {memory_profiler.__version__}")