import numpy as np
from benchmark import benchmark

from mops.reference_implementations import homogeneous_polynomial_evaluation as ref_hpe
from mops import homogeneous_polynomial_evaluation as hpe

np.random.seed(0xDEADBEEF)

A = np.random.rand(1000, 300)
C = np.random.rand(2000)
P = np.random.randint(300, size=(2000, 4))

ref_mean, ref_std = benchmark(lambda: ref_hpe(A, C, P))
mean, std = benchmark(lambda: hpe(A, C, P))

print(f"{ref_mean:.2e}", f"{ref_std:.2e}")
print(f"{mean:.2e}", f"{std:.2e}")

print("Speed-up:", ref_mean/mean)
