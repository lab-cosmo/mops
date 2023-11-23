import numpy as np
from benchmark import benchmark, format_mean_std

from mops import sparse_accumulation_scatter_add_with_weights as sasaw

# from mops.reference_implementations import (
#     sparse_accumulation_scatter_add_with_weights as ref_sasaw,
# )

np.random.seed(0xDEADBEEF)

A = np.random.rand(60000, 13)
B = np.random.rand(60000, 32)
W = np.random.rand(1000, 7, 32)
C = np.random.rand(900)
indices_output_1 = np.random.randint(1000, size=(60000,))
indices_W_1 = np.random.randint(1000, size=(60000,))
output_size = 100
indices_A = np.random.randint(13, size=(900,))
indices_W_2 = np.random.randint(7, size=(900,))
indices_output_2 = np.random.randint(output_size, size=(900,))

# ref_mean, ref_std = benchmark(
#     lambda: ref_sasaw(A, B, C, W, indices_A, indices_W_1, indices_W_2, indices_output_1, indices_output_2, output_size), repeats=3, warmup=1
# )  # very slow
mean, std = benchmark(
    lambda: sasaw(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        output_size,
    )
)

# print("Beference implementation:", format_mean_std(ref_mean, ref_std))
print("Optimized implementation:", format_mean_std(mean, std))

# print("Speed-up:", ref_mean / mean)
