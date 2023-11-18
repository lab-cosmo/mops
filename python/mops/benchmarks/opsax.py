import numpy as np
from benchmark import benchmark, format_mean_std

from mops import outer_product_scatter_add_with_weights as opsax

# from mops.reference_implementations import (
#     outer_product_scatter_add_with_weights as ref_opsax,
# )


np.random.seed(0xDEADBEEF)


A = np.random.rand(60000, 13)
R = np.random.rand(60000, 32)
X = np.random.rand(1000, 32)

I = np.sort(np.random.randint(1000, size=(60000,)))
J = np.random.randint(1000, size=(60000,))

# ref_mean, ref_std = benchmark(lambda: ref_opsax(A, R, X, I, J))
mean, std = benchmark(lambda: opsax(A, R, X, I, J))

# print("Reference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

# print("Speed-up:", ref_mean / mean)
