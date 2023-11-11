import numpy as np
from benchmark import benchmark, format_mean_std
from mops.reference_implementations import (
    outer_product_scatter_add_with_weights as ref_opsax,
)

from mops import outer_product_scatter_add_with_weights as opsax

np.random.seed(0xDEADBEEF)


A = np.random.rand(100, 10)
R = np.random.rand(100, 5)
X = np.random.rand(20, 5)

I = np.random.randint(20, size=(100,))
J = np.random.randint(20, size=(100,))

ref_mean, ref_std = benchmark(lambda: ref_opsax(A, R, X, I, J))
mean, std = benchmark(lambda: opsax(A, R, X, I, J))

print("Reference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

print("Speed-up:", ref_mean / mean)
