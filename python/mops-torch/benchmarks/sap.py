import mops.torch
import torch
from benchmark import benchmark, format_mean_std

torch.manual_seed(0xDEADBEEF)


A = torch.rand(1000, 20, requires_grad=True)
B = torch.rand(1000, 5, requires_grad=True)
C = torch.rand(100)

P_A = torch.sort(torch.randint(20, size=(100,), dtype=torch.int32))[0]
P_B = torch.sort(torch.randint(5, size=(100,), dtype=torch.int32))[0]
n_O = 50
P_O = torch.sort(torch.randint(n_O, size=(100,), dtype=torch.int32))[0]

mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(
        mops.torch.sparse_accumulation_of_products(A, B, C, P_A, P_B, P_O, n_O)
    ),
    repeats=10,
    warmup=10,
)

print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
