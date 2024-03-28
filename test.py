from mops.reference_implementations import outer_product_scatter_add as ref_opsa
from mops import outer_product_scatter_add as opsa

import cupy as cp


A = cp.random.rand(100, 4)
B = cp.random.rand(100, 5)
indices_output = cp.random.randint(10, size=100)
indices_output = cp.sort(indices_output)
output_size = 20  # output_size = 10 works well

output = opsa(A, B, indices_output, output_size)
ref_output = ref_opsa(A, B, indices_output, output_size)

assert cp.allclose(output, ref_output)
