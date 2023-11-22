import torch

from ._c_lib import _lib_path

torch.ops.load_library(_lib_path())

outer_product_scatter_add = torch.ops.mops.outer_product_scatter_add
