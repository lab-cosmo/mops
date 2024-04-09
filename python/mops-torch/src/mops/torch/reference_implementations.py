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
    output = torch.zeros(A.shape[0], output_size, device=A.device)
    output.index_add_(1, indices_output, product)

    return output


def outer_product_scatter_add_with_weights(A, B, W, indices_W, indices_output):
    output = torch.zeros(W.shape[0], A.shape[1], B.shape[1], device=A.device)
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
    output_size,
):
    # ???
    pass
