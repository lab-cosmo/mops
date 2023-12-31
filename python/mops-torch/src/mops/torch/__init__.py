import torch

from ._c_lib import _lib_path

torch.ops.load_library(_lib_path())

homogeneous_polynomial_evaluation = torch.ops.mops.homogeneous_polynomial_evaluation
outer_product_scatter_add = torch.ops.mops.outer_product_scatter_add
sparse_accumulation_of_products = torch.ops.mops.sparse_accumulation_of_products
outer_product_scatter_add_with_weights = (
    torch.ops.mops.outer_product_scatter_add_with_weights
)
sparse_accumulation_scatter_add_with_weights = (
    torch.ops.mops.sparse_accumulation_scatter_add_with_weights
)
