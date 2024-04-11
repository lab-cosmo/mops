import torch


def homogeneous_polynomial_evaluation(A, C, idx):
    indexed_A = A[:, idx]
    monomials = torch.prod(indexed_A, dim=2)
    return torch.sum(C * monomials, dim=1)


def outer_product_scatter_add(A, B, idx, n_out: int):
    outer = A.unsqueeze(2) * B.unsqueeze(1)
    out = torch.zeros(n_out, A.shape[1], B.shape[1], device=A.device, dtype=A.dtype)
    out.index_add_(0, idx, outer)
    return out


def sparse_accumulation_of_products(
    A, B, C, indices_A, indices_B, indices_output, output_size: int
):
    sparse_A = A[:, indices_A]
    sparse_B = B[:, indices_B]
    product = C * sparse_A * sparse_B
    output = torch.zeros(A.shape[0], output_size, device=A.device, dtype=A.dtype)
    output.index_add_(1, indices_output, product)

    return output


def outer_product_scatter_add_with_weights(A, B, W, indices_W, indices_output):
    output = torch.zeros(
        W.shape[0], A.shape[1], B.shape[1], device=A.device, dtype=A.dtype
    )
    indexed_W = W[indices_W]
    ABW = torch.einsum("ea,eb,eb->eab", A, B, indexed_W)
    output.index_add_(0, indices_output, ABW)
    return output


def sparse_accumulation_scatter_add_with_weights(
    A,
    B,
    C,
    W,
    indices_A,
    indices_W_1,
    indices_W_2,
    indices_output_1,
    indices_output_2,
    output_size: int,
):
    indexed_W_1 = W[indices_W_1]
    AB = torch.einsum("ea,eb->eab", A, B)
    AB = AB.swapaxes(1, 2).reshape(-1, A.shape[1])
    indexed_W_1 = indexed_W_1.swapaxes(1, 2).reshape(-1, indexed_W_1.shape[1])
    ABW = sparse_accumulation_of_products(
        AB, indexed_W_1, C, indices_A, indices_W_2, indices_output_2, output_size
    )
    ABW = ABW.reshape(A.shape[0], -1, output_size).swapaxes(1, 2)
    output = torch.zeros(
        W.shape[0], output_size, ABW.shape[-1], device=A.device, dtype=A.dtype
    )
    output.index_add_(0, indices_output_1, ABW)
    return output
