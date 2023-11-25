import numpy as np
from benchmark import benchmark, format_mean_std
from mops.reference_implementations import (
    outer_product_scatter_add_with_weights as ref_opsaw,
)

from mops import outer_product_scatter_add_with_weights as opsaw

np.random.seed(0xDEADBEEF)


A = np.random.rand(60000, 13)
B = np.random.rand(60000, 32)
W = np.random.rand(1000, 32)

indices_W = np.random.randint(1000, size=(60000,))
indices_output = np.random.randint(1000, size=(60000,))

ref_mean, ref_std = benchmark(lambda: ref_opsaw(A, B, W, indices_W, indices_output))
mean, std = benchmark(lambda: opsaw(A, B, W, indices_W, indices_output))

print("Beference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

print("Speed-up:", ref_mean / mean)
