import torch


def outer_product_scatter_add(A, B, idx, n_out: int):
    outer = A.unsqueeze(2) * B.unsqueeze(1)
    out = torch.zeros(n_out, A.shape[1], B.shape[1], device=A.device, dtype=A.dtype)
    out.index_add_(0, idx, outer)
    return out
