import mops.torch
import torch
from benchmark import benchmark, format_mean_std, initialize
from mops.torch.reference_implementations import (
    homogeneous_polynomial_evaluation as ref_hpe,
)

initialize()

A = torch.rand(1000, 2000, requires_grad=True)
C = torch.rand(100000)
indices_A = torch.randint(2000, size=(100000, 4), dtype=torch.int32)

ref_hpe = torch.jit.script(ref_hpe)
mean_fwd_ref, std_fwd_ref, mean_bwd_ref, std_bwd_ref = benchmark(
    lambda: torch.sum(ref_hpe(A, C, indices_A))
)


def hpe(A, C, indices_A):
    return mops.torch.homogeneous_polynomial_evaluation(A, C, indices_A)


hpe = torch.jit.script(hpe)
mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(hpe(A, C, indices_A))
)

print()
print("Reference implementation:")
print("Forward pass:", format_mean_std(mean_fwd_ref, std_fwd_ref))
print("Backward pass:", format_mean_std(mean_bwd_ref, std_bwd_ref))

print()
print("Optimized implementation:")
print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
