import mops.torch
import torch
from benchmark import benchmark, format_mean_std

torch.manual_seed(0xDEADBEEF)


A = torch.rand(1000, 2000, requires_grad=True)
C = torch.rand(100000)
indices_A = torch.randint(2000, size=(100000, 4), dtype=torch.int32)


mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(mops.torch.homogeneous_polynomial_evaluation(A, C, indices_A))
)

print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
