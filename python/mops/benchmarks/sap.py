import numpy as np
from benchmark import benchmark, format_mean_std
from mops.reference_implementations import sparse_accumulation_of_products as ref_sap

from mops import sparse_accumulation_of_products as sap

np.random.seed(0xDEADBEEF)

A = np.random.rand(1000, 20)
B = np.random.rand(1000, 6)
C = np.random.rand(100)

P_A = np.random.randint(20, size=(100,))
P_B = np.random.randint(6, size=(100,))
n_O = 50
P_O = np.random.randint(n_O, size=(100,))

ref_mean, ref_std = benchmark(lambda: ref_sap(A, B, C, P_A, P_B, P_O, n_O))
mean, std = benchmark(lambda: sap(A, B, C, P_A, P_B, P_O, n_O))

print("Reference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

print("Speed-up:", ref_mean / mean)
