import numpy as np

# Load data (assumes 2 columns: x y)
data = np.loadtxt('airfoil.txt')

# Validate shape
if data.ndim != 2 or data.shape[1] != 2:
    raise ValueError("File must contain exactly two columns: x y")

# Reverse order of points
data = data[::-1]

# Write back to the same file
np.savetxt('airfoil.txt', data, fmt="%.8f")