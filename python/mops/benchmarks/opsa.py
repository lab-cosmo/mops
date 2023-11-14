import numpy as np
from benchmark import benchmark, format_mean_std
from mops.reference_implementations import outer_product_scatter_add as ref_opsa

from mops import outer_product_scatter_add as opsa

np.random.seed(0xDEADBEEF)

A = np.random.rand(60000, 13)
B = np.random.rand(60000, 20)

indices = np.sort(np.random.randint(1000, size=(60000,)))

ref_mean, ref_std = benchmark(lambda: ref_opsa(A, B, indices, np.max(indices) + 1))
mean, std = benchmark(lambda: opsa(A, B, indices, np.max(indices) + 1))

print("Reference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

print("Speed-up:", ref_mean / mean)
