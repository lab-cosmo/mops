import mops.torch
import torch
from benchmark import benchmark, format_mean_std

torch.manual_seed(0xDEADBEEF)


A = torch.rand(32000, 13, requires_grad=True)
B = torch.rand(32000, 7, requires_grad=True)
C = torch.rand(900)
indices_A = torch.randint(13, size=(900,), dtype=torch.int32)
indices_B = torch.randint(7, size=(900,), dtype=torch.int32)
n_output = 100
indices_output = torch.randint(n_output, size=(900,), dtype=torch.int32)

mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(
        mops.torch.sparse_accumulation_of_products(
            A, B, C, indices_A, indices_B, indices_output, n_output
        )
    )
)

print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
