import mops.torch
import torch
from benchmark import benchmark, format_mean_std, initialize
from mops.torch.reference_implementations import (
    sparse_accumulation_of_products as ref_sap,
)

initialize()

A = torch.rand(32000, 13, requires_grad=True)
B = torch.rand(32000, 7, requires_grad=True)
C = torch.rand(900)
indices_A = torch.randint(13, size=(900,), dtype=torch.int32)
indices_B = torch.randint(7, size=(900,), dtype=torch.int32)
n_output = 100
indices_output = torch.randint(n_output, size=(900,), dtype=torch.int32)

ref_sap = torch.jit.script(ref_sap)
mean_fwd_ref, std_fwd_ref, mean_bwd_ref, std_bwd_ref = benchmark(
    lambda: torch.sum(ref_sap(A, B, C, indices_A, indices_B, indices_output, n_output))
)


def sap(A, B, C, indices_A, indices_B, indices_output, n_output: int):
    return mops.torch.sparse_accumulation_of_products(
        A, B, C, indices_A, indices_B, indices_output, n_output
    )


sap = torch.jit.script(sap)
mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(sap(A, B, C, indices_A, indices_B, indices_output, n_output))
)

print()
print("Reference implementation:")
print("Forward pass:", format_mean_std(mean_fwd_ref, std_fwd_ref))
print("Backward pass:", format_mean_std(mean_bwd_ref, std_bwd_ref))

print()
print("Optimized implementation:")
print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
