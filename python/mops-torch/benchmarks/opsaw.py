import mops.torch
import torch
from benchmark import benchmark, format_mean_std

torch.manual_seed(0xDEADBEEF)


A = torch.rand(60000, 13, requires_grad=True)
B = torch.rand(60000, 32, requires_grad=True)
W = torch.rand(1000, 32, requires_grad=True)
indices_W = torch.randint(1000, size=(60000,), dtype=torch.int32)
indices_output = torch.randint(1000, size=(60000,), dtype=torch.int32)

mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(
        mops.torch.outer_product_scatter_add_with_weights(
            A, B, W, indices_W, indices_output
        )
    )
)

print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
