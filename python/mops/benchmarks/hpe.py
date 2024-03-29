import numpy as np
from benchmark import benchmark, format_mean_std

from mops import homogeneous_polynomial_evaluation as hpe

np.random.seed(0xDEADBEEF)

A = np.random.rand(1000, 2000)
C = np.random.rand(100000)
indices_A = np.random.randint(2000, size=(100000, 4))

# ref_mean, ref_std = benchmark(lambda: ref_hpe(A, C, indices_A))
mean, std = benchmark(lambda: hpe(A, C, indices_A))

# print("Reference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

# print("Speed-up:", ref_mean / mean)
