import mops.torch
import torch
from benchmark import benchmark, format_mean_std, initialize
from mops.torch.reference_implementations import (
    sparse_accumulation_scatter_add_with_weights as ref_sasaw,
)

initialize()

A = torch.rand(60000, 13, requires_grad=True)
B = torch.rand(60000, 32, requires_grad=True)
C = torch.rand(900)
W = torch.rand(1000, 7, 32, requires_grad=True)
indices_output_1 = torch.randint(1000, size=(60000,), dtype=torch.int32)
indices_W_1 = torch.randint(1000, size=(60000,), dtype=torch.int32)
output_size = 100
indices_A = torch.randint(13, size=(900,), dtype=torch.int32)
indices_W_2 = torch.randint(7, size=(900,), dtype=torch.int32)
indices_output_2 = torch.randint(output_size, size=(900,), dtype=torch.int32)

ref_sasaw = torch.jit.script(ref_sasaw)
mean_fwd_ref, std_fwd_ref, mean_bwd_ref, std_bwd_ref = benchmark(
    lambda: torch.sum(
        ref_sasaw(
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_W_2,
            indices_output_1,
            indices_output_2,
            output_size,
        )
    )
)


def sasaw(
    A,
    B,
    C,
    W,
    indices_A,
    indices_W_1,
    indices_output_1,
    indices_W_2,
    indices_output_2,
    output_size: int,
):
    return mops.torch.sparse_accumulation_scatter_add_with_weights(
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_output_1,
        indices_W_2,
        indices_output_2,
        output_size,
    )


sasaw = torch.jit.script(sasaw)
mean_fwd, std_fwd, mean_bwd, std_bwd = benchmark(
    lambda: torch.sum(
        sasaw(
            A,
            B,
            C,
            W,
            indices_A,
            indices_W_1,
            indices_output_1,
            indices_W_2,
            indices_output_2,
            output_size,
        )
    )
)

print()
print("Reference implementation:")
print("Forward pass:", format_mean_std(mean_fwd_ref, std_fwd_ref))
print("Backward pass:", format_mean_std(mean_bwd_ref, std_bwd_ref))

print()
print("Optimized implementation:")
print("Forward pass:", format_mean_std(mean_fwd, std_fwd))
print("Backward pass:", format_mean_std(mean_bwd, std_bwd))
