import mops.torch
import torch
from benchmark import benchmark, format_mean_std

torch.manual_seed(0xDEADBEEF)


A = torch.rand(1000, 20, requires_grad=True)
B = torch.rand(1000, 5, requires_grad=True)

indices = torch.sort(torch.randint(10, size=(1000,), dtype=torch.int32))[0]

mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(mops.torch.outer_product_scatter_add(A, B, indices, 10)),
    repeats=10,
    warmup=10,
)

print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
