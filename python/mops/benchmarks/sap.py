import numpy as np
from benchmark import benchmark, format_mean_std

from mops import sparse_accumulation_of_products as sap

# from mops.reference_implementations import sparse_accumulation_of_products as ref_sap


np.random.seed(0xDEADBEEF)

A = np.random.rand(32000, 13)
B = np.random.rand(32000, 7)
C = np.random.rand(900)

P_A = np.random.randint(13, size=(900,))
P_B = np.random.randint(7, size=(900,))
n_O = 100
P_O = np.sort(np.random.randint(n_O, size=(900,)))

# ref_mean, ref_std = benchmark(lambda: ref_sap(A, B, C, P_A, P_B, P_O, n_O))
mean, std = benchmark(lambda: sap(A, B, C, P_A, P_B, P_O, n_O))

# print("Reference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

# print("Speed-up:", ref_mean / mean)
