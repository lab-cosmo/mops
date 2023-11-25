import mops.torch
import torch
from benchmark import benchmark, format_mean_std

torch.manual_seed(0xDEADBEEF)


A = torch.rand(1000, 20, requires_grad=True)
R = torch.rand(1000, 5, requires_grad=True)
X = torch.rand(50, 5, requires_grad=True)

I = torch.randint(20, size=(100,), dtype=torch.int32)
J = torch.randint(20, size=(100,), dtype=torch.int32)

mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(mops.torch.outer_product_scatter_add_with_weights(A, R, X, I, J)),
    repeats=10,
    warmup=10,
)

print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
