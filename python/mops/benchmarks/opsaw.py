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
