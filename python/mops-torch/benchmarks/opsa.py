import mops.torch
import torch
from benchmark import benchmark, format_mean_std, initialize
from mops.torch.reference_implementations import outer_product_scatter_add as ref_opsa

initialize()

A = torch.rand(1000, 20, requires_grad=True)
B = torch.rand(1000, 5, requires_grad=True)
output_size = 1000
indices = torch.sort(torch.randint(10, size=(1000,), dtype=torch.int32)).values

ref_opsa = torch.jit.script(ref_opsa)
mean_fwd_ref, std_fwd_ref, mean_bwd_ref, std_bwd_ref = benchmark(
    lambda: torch.sum(ref_opsa(A, B, indices, output_size))
)


def opsa(A, B, indices, output_size: int):
    return mops.torch.outer_product_scatter_add(A, B, indices, output_size)


opsa = torch.jit.script(opsa)
mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(opsa(A, B, indices, output_size))
)

print()
print("Reference implementation:")
print("Forward pass:", format_mean_std(mean_fwd_ref, std_fwd_ref))
print("Backward pass:", format_mean_std(mean_bwd_ref, std_bwd_ref))

print()
print("Optimized implementation:")
print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
