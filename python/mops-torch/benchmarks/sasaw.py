import mops.torch
import torch
from benchmark import benchmark, format_mean_std

torch.manual_seed(0xDEADBEEF)


A = torch.rand(100, 20, requires_grad=True)
R = torch.rand(100, 200, requires_grad=True)
X = torch.rand(25, 13, 200, requires_grad=True)
C = torch.rand(50)
I = torch.randint(25, size=(100,), dtype=torch.int32)
J = torch.randint(25, size=(100,), dtype=torch.int32)
n_O2 = 15
M_1 = torch.randint(20, size=(50,), dtype=torch.int32)
M_2 = torch.randint(13, size=(50,), dtype=torch.int32)
M_3 = torch.randint(n_O2, size=(50,), dtype=torch.int32)

mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(
        mops.torch.sparse_accumulation_scatter_add_with_weights(
            A, R, C, X, M_1, J, M_2, I, M_3, n_O2
        )
    ),
    repeats=10,
    warmup=10,
)

print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
