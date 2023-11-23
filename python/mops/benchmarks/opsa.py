import numpy as np
from benchmark import benchmark, format_mean_std

from mops import outer_product_scatter_add as opsa

# from mops.reference_implementations import outer_product_scatter_add as ref_opsa


np.random.seed(0xDEADBEEF)

A = np.random.rand(60000, 13)
B = np.random.rand(60000, 20)

output_size = 1000
indices_output = np.sort(np.random.randint(output_size, size=(60000,)))

# ref_mean, ref_std = benchmark(lambda: ref_opsa(A, B, indices_output, output_size))
mean, std = benchmark(lambda: opsa(A, B, indices_output, output_size))

# print("Reference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

# print("Speed-up:", ref_mean / mean)
