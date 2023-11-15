import numpy as np
from benchmark import benchmark, format_mean_std

from mops import sparse_accumulation_scatter_add_with_weights as sasax

# from mops.reference_implementations import (
#     sparse_accumulation_scatter_add_with_weights as ref_sasax,
# )

np.random.seed(0xDEADBEEF)

A = np.random.rand(60000, 13)
R = np.random.rand(60000, 32)
X = np.random.rand(1000, 7, 32)
C = np.random.rand(900)
I = np.sort(np.random.randint(1000, size=(60000,)))
J = np.random.randint(1000, size=(60000,))
n_O = 100
M_1 = np.random.randint(13, size=(900,))
M_2 = np.random.randint(7, size=(900,))
M_3 = np.random.randint(n_O, size=(900,))

# ref_mean, ref_std = benchmark(
#     lambda: ref_sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O), repeats=3, warmup=1
# )  # very slow
mean, std = benchmark(lambda: sasax(A, R, X, C, I, J, M_1, M_2, M_3, n_O))

# print("Reference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

# print("Speed-up:", ref_mean / mean)
