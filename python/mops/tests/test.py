import mops
import numpy as np
import cupy as cp
import mops.reference_implementations


A = np.array([[1, 2], [3, 4]], dtype=np.float32)
B = np.array([[1, 2], [3, 4]], dtype=np.float32)
indices_output = np.array([2, 2], dtype=np.int32)
output_size = 3

reference = mops.reference_implementations.outer_product_scatter_add(A, B, indices_output, output_size)
print(reference)

actual = mops.outer_product_scatter_add(cp.array(A), cp.array(B), cp.array(indices_output), output_size)
print(actual)
