import mops.torch
import torch


def test_hpe_torchscript():

    A_torch = torch.rand(100, 20, requires_grad=True)
    C_torch = torch.rand(200)
    indices_A_torch = torch.randint(20, size=(200, 4), dtype=torch.int32)

    A_torchscript = A_torch.clone().detach().requires_grad_(True)
    C_torchscript = C_torch.clone().detach()
    indices_A_torchscript = indices_A_torch.clone().detach()

    result_torch = mops.torch.homogeneous_polynomial_evaluation(
        A_torch, C_torch, indices_A_torch
    )
    torch.sum(result_torch).backward()

    def fn(A, C, indices):
        return mops.torch.homogeneous_polynomial_evaluation(A, C, indices)

    scripted_fn = torch.jit.script(fn)
    result_torchscript = scripted_fn(
        A_torchscript, C_torchscript, indices_A_torchscript
    )
    torch.sum(result_torchscript).backward()

    assert torch.allclose(result_torch, result_torchscript)
    assert torch.allclose(A_torch.grad, A_torchscript.grad)


def test_opsa_torchscript():

    A_torch = torch.rand(100, 20, requires_grad=True)
    B_torch = torch.rand(100, 5, requires_grad=True)
    output_size = 10
    indices_output_torch = torch.randint(output_size, size=(100,), dtype=torch.int32)

    A_torchscript = A_torch.clone().detach().requires_grad_(True)
    B_torchscript = B_torch.clone().detach().requires_grad_(True)
    indices_output_torchscript = indices_output_torch.clone().detach()

    result_torch = mops.torch.outer_product_scatter_add(
        A_torch, B_torch, indices_output_torch, output_size
    )
    torch.sum(result_torch).backward()

    def fn(A, B, indices, output_size: int):
        return mops.torch.outer_product_scatter_add(A, B, indices, output_size)

    scripted_fn = torch.jit.script(fn)
    result_torchscript = scripted_fn(
        A_torchscript, B_torchscript, indices_output_torchscript, output_size
    )
    torch.sum(result_torchscript).backward()

    assert torch.allclose(result_torch, result_torchscript)
    assert torch.allclose(A_torch.grad, A_torchscript.grad)
    assert torch.allclose(B_torch.grad, B_torchscript.grad)


def test_sap_torchscript():

    A_torch = torch.rand(100, 20, requires_grad=True)
    B_torch = torch.rand(100, 6, requires_grad=True)
    C_torch = torch.rand(30)
    indices_A_torch = torch.randint(20, size=(30,), dtype=torch.int32)
    indices_B_torch = torch.randint(6, size=(30,), dtype=torch.int32)
    output_size = 35
    indices_output_torch = torch.randint(output_size, size=(30,), dtype=torch.int32)

    A_torchscript = A_torch.clone().detach().requires_grad_(True)
    B_torchscript = B_torch.clone().detach().requires_grad_(True)
    C_torchscript = C_torch.clone().detach()
    indices_A_torchscript = indices_A_torch.clone().detach()
    indices_B_torchscript = indices_B_torch.clone().detach()
    indices_output_torchscript = indices_output_torch.clone().detach()

    result_torch = mops.torch.sparse_accumulation_of_products(
        A_torch,
        B_torch,
        C_torch,
        indices_A_torch,
        indices_B_torch,
        indices_output_torch,
        output_size,
    )
    torch.sum(result_torch).backward()

    def fn(A, B, C, indices_A, indices_B, indices_output, output_size: int):
        return mops.torch.sparse_accumulation_of_products(
            A, B, C, indices_A, indices_B, indices_output, output_size
        )

    scripted_fn = torch.jit.script(fn)
    result_torchscript = scripted_fn(
        A_torchscript,
        B_torchscript,
        C_torchscript,
        indices_A_torchscript,
        indices_B_torchscript,
        indices_output_torchscript,
        output_size,
    )
    torch.sum(result_torchscript).backward()

    assert torch.allclose(result_torch, result_torchscript)
    assert torch.allclose(A_torch.grad, A_torchscript.grad)
    assert torch.allclose(B_torch.grad, B_torchscript.grad)


def test_opsaw_torchscript():

    A_torch = torch.rand(100, 10, requires_grad=True)
    B_torch = torch.rand(100, 5, requires_grad=True)
    W_torch = torch.rand(20, 5, requires_grad=True)
    indices_W_torch = torch.randint(20, size=(100,), dtype=torch.int32)
    indices_output_torch = torch.randint(20, size=(100,), dtype=torch.int32)

    A_torchscript = A_torch.clone().detach().requires_grad_(True)
    B_torchscript = B_torch.clone().detach().requires_grad_(True)
    W_torchscript = W_torch.clone().detach().requires_grad_(True)
    indices_W_torchscript = indices_W_torch.clone().detach()
    indices_output_torchscript = indices_output_torch.clone().detach()

    result_torch = mops.torch.outer_product_scatter_add_with_weights(
        A_torch, B_torch, W_torch, indices_W_torch, indices_output_torch
    )
    torch.sum(result_torch).backward()

    def fn(A, B, W, indices_W, indices_output):
        return mops.torch.outer_product_scatter_add_with_weights(
            A, B, W, indices_W, indices_output
        )

    scripted_fn = torch.jit.script(fn)
    result_torchscript = scripted_fn(
        A_torchscript,
        B_torchscript,
        W_torchscript,
        indices_W_torchscript,
        indices_output_torchscript,
    )
    torch.sum(result_torchscript).backward()

    assert torch.allclose(result_torch, result_torchscript)
    assert torch.allclose(A_torch.grad, A_torchscript.grad)
    assert torch.allclose(B_torch.grad, B_torchscript.grad)
    assert torch.allclose(W_torch.grad, W_torchscript.grad)


def test_sasaw_torchscript():

    A_torch = torch.rand(100, 20, requires_grad=True)
    B_torch = torch.rand(100, 200, requires_grad=True)
    W_torch = torch.rand(25, 13, 200, requires_grad=True)
    C_torch = torch.rand(50)
    indices_output_1_torch = torch.randint(25, size=(100,), dtype=torch.int32)
    indices_W_1_torch = torch.randint(25, size=(100,), dtype=torch.int32)
    output_size = 15
    indices_A_torch = torch.randint(20, size=(50,), dtype=torch.int32)
    indices_W_2_torch = torch.randint(13, size=(50,), dtype=torch.int32)
    indices_output_2_torch = torch.randint(output_size, size=(50,), dtype=torch.int32)

    A_torchscript = A_torch.clone().detach().requires_grad_(True)
    B_torchscript = B_torch.clone().detach().requires_grad_(True)
    W_torchscript = W_torch.clone().detach().requires_grad_(True)
    C_torchscript = C_torch.clone().detach()
    indices_A_torchscript = indices_A_torch.clone().detach()
    indices_W_1_torchscript = indices_W_1_torch.clone().detach()
    indices_W_2_torchscript = indices_W_2_torch.clone().detach()
    indices_output_1_torchscript = indices_output_1_torch.clone().detach()
    indices_output_2_torchscript = indices_output_2_torch.clone().detach()

    result_torch = mops.torch.sparse_accumulation_scatter_add_with_weights(
        A_torch,
        B_torch,
        C_torch,
        W_torch,
        indices_A_torch,
        indices_W_1_torch,
        indices_W_2_torch,
        indices_output_1_torch,
        indices_output_2_torch,
        output_size,
    )
    torch.sum(result_torch).backward()

    def fn(
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
        return mops.torch.sparse_accumulation_scatter_add_with_weights(
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

    scripted_fn = torch.jit.script(fn)
    result_torchscript = scripted_fn(
        A_torchscript,
        B_torchscript,
        C_torchscript,
        W_torchscript,
        indices_A_torchscript,
        indices_W_1_torchscript,
        indices_W_2_torchscript,
        indices_output_1_torchscript,
        indices_output_2_torchscript,
        output_size,
    )
    torch.sum(result_torchscript).backward()

    assert torch.allclose(result_torch, result_torchscript)
    assert torch.allclose(A_torch.grad, A_torchscript.grad)
    assert torch.allclose(B_torch.grad, B_torchscript.grad)
    assert torch.allclose(W_torch.grad, W_torchscript.grad)
