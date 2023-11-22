import numpy as np
import pytest

import mops
from mops import reference_implementations as ref

np.random.seed(0xDEADBEEF)


def test_sap():
    A = np.random.rand(100, 20)
    B = np.random.rand(100, 6)
    C = np.random.rand(30)

    indices_A = np.random.randint(20, size=(30,))
    indices_B = np.random.randint(6, size=(30,))
    output_size = 35
    indices_output = np.random.randint(output_size, size=(30,))

    reference = ref.sparse_accumulation_of_products(
        A, B, C, indices_A, indices_B, indices_output, output_size
    )
    actual = mops.sparse_accumulation_of_products(
        A, B, C, indices_A, indices_B, indices_output, output_size
    )
    assert np.allclose(reference, actual)


def test_sap_wrong_type():
    message = (
        "`A` must be a 2D array in sparse_accumulation_of_products, "
        "got a 1D array instead"
    )
    with pytest.raises(ValueError, match=message):
        mops.sparse_accumulation_of_products(np.array(1), 2, 3, 4, 5, 6, 7)
