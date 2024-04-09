import mops.torch
import torch
from benchmark import benchmark, format_mean_std, initialize
from mops.torch.reference_implementations import (
    homogeneous_polynomial_evaluation as ref_hpe,
)

initialize()


A = torch.rand(60000, 13, requires_grad=True)
B = torch.rand(60000, 32, requires_grad=True)
W = torch.rand(1000, 32, requires_grad=True)
indices_W = torch.randint(1000, size=(60000,), dtype=torch.int32)
indices_output = torch.randint(1000, size=(60000,), dtype=torch.int32)

ref_hpe = torch.jit.script(ref_hpe)
mean_fwd_ref, std_fwd_ref, mean_bwd_ref, std_bwd_ref = benchmark(
    lambda: torch.sum(ref_hpe(A, B, W, indices_W, indices_output))
)


def hpe(A, B, W, indices_W, indices_output):
    return mops.torch.homogeneous_polynomial_evaluation(
        A, B, W, indices_W, indices_output
    )


hpe = torch.jit.script(hpe)
mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(hpe(A, B, W, indices_W, indices_output))
)

print()
print("Reference implementation:")
print("Forward pass:", format_mean_std(mean_fwd_ref, std_fwd_ref))
print("Backward pass:", format_mean_std(mean_bwd_ref, std_bwd_ref))

print()
print("Optimized implementation:")
print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
