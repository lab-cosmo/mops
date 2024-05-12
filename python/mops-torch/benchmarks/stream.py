import mops.torch
import torch
from benchmark import benchmark, format_mean_std, initialize
from typing import List

initialize()

A = torch.rand(100, 1000, requires_grad=True)
C = torch.rand(2000)
indices_A = torch.randint(1000, size=(2000, 3), dtype=torch.int32)

@torch.jit.script
def hpe_10(A, C, indices_A):
    results = []
    for _ in range(10):
        results.append(mops.torch.homogeneous_polynomial_evaluation(A, C, indices_A))
    return torch.sum(torch.concatenate(results))

mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: hpe_10(A, C, indices_A)
)


@torch.jit.script
def hpe_10_stream(streams: List[torch.classes.cuda.Stream], A, C, indices_A):
    results = []
    
    for stream in streams:
        with torch.cuda.stream(stream):
            result = mops.torch.homogeneous_polynomial_evaluation(A, C, indices_A)
        results.append(result)

    for stream in streams:
        stream.synchronize()

    return torch.sum(torch.concatenate(results))

streams = [torch.classes.cuda.Stream() for _ in range(10)]
mean_fwd_stream, std_fwd_stream, mean_bwd_stream, std_bwd_stream = benchmark(
    lambda: hpe_10_stream(streams, A, C, indices_A)
)


print()
print("Without CUDA streams:")
print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))

print()
print("With CUDA streams:")
print("Forward pass:", format_mean_std(mean_fwd_stream, std_fwd_stream))
print("Backward pass:", format_mean_std(mean_bwd_stream, std_bwd_stream))
