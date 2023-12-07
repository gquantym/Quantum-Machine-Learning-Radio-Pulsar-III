import numpy as np

# Assuming arr1 and arr2 are your 3D arrays
arr1 = np.random.rand(5, 10, 3)
arr2 = np.random.rand(5, 10, 3)

# Vertically stack along the first axis (axis=0)
result = np.concatenate((arr1, arr2), axis=1)

print(result.shape)
