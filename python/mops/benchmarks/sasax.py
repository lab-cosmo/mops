import numpy as np
from benchmark import benchmark, format_mean_std
from mops.reference_implementations import (
    sparse_accumulation_scatter_add_with_weights as ref_sasax,
)

from mops import sparse_accumulation_scatter_add_with_weights as sasax

np.random.seed(0xDEADBEEF)

A = np.random.rand(100, 20)
R = np.random.rand(100, 200)
X = np.random.rand(25, 13, 200)
C = np.random.rand(50)
n_O1 = 25
I = np.random.randint(25, size=(100,))
J = np.random.randint(25, size=(100,))
n_O2 = 15
M_1 = np.random.randint(20, size=(50,))
M_2 = np.random.randint(13, size=(50,))
M_3 = np.random.randint(n_O2, size=(50,))

ref_mean, ref_std = benchmark(
    lambda: ref_sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O1, n_O2)
)
mean, std = benchmark(lambda: sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O1, n_O2))

print("Reference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

print("Speed-up:", ref_mean / mean)
