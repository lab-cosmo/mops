import numpy as np
from benchmark import benchmark, format_mean_std
from mops.reference_implementations import (
    outer_product_scatter_add_with_weights as ref_opsaw,
)

from mops import outer_product_scatter_add_with_weights as opsaw

np.random.seed(0xDEADBEEF)


A = np.random.rand(100, 10)
R = np.random.rand(100, 5)
n_O = 20
X = np.random.rand(n_O, 5)

I = np.random.randint(20, size=(100,))
J = np.random.randint(20, size=(100,))

ref_mean, ref_std = benchmark(lambda: ref_opsaw(A, R, X, I, J, 20))
mean, std = benchmark(lambda: opsaw(A, R, X, I, J, 20))

print("Reference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

print("Speed-up:", ref_mean / mean)
