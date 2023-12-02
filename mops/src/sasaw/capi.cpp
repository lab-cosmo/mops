#include "mops/capi.hpp"

#include "mops/sasaw.h"
#include "mops/sasaw.hpp"

static size_t checked_cast(int64_t value) {
    if (value < 0 ||
        static_cast<uint64_t>(value) > static_cast<uint64_t>(SIZE_MAX)) {
        throw std::runtime_error("integer value '" + std::to_string(value) +
                                 "' does not fit in size_t");
    }
    return static_cast<size_t>(value);
}

extern "C" int mops_sparse_accumulation_scatter_add_with_weights_f32(
    mops_tensor_3d_f32_t output, mops_tensor_2d_f32_t A, mops_tensor_2d_f32_t B,
    mops_tensor_1d_f32_t C, mops_tensor_3d_f32_t W,
    mops_tensor_1d_i32_t indices_A, mops_tensor_1d_i32_t indices_W_1,
    mops_tensor_1d_i32_t indices_W_2, mops_tensor_1d_i32_t indices_output_1,
    mops_tensor_1d_i32_t indices_output_2) {
    MOPS_CATCH_EXCEPTIONS(
        mops::sparse_accumulation_scatter_add_with_weights<float>(
            {output.data,
             {checked_cast(output.shape[0]), checked_cast(output.shape[1]),
              checked_cast(output.shape[2])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {W.data,
             {checked_cast(W.shape[0]), checked_cast(W.shape[1]),
              checked_cast(W.shape[2])}},
            {indices_A.data, {checked_cast(indices_A.shape[0])}},
            {indices_W_1.data, {checked_cast(indices_W_1.shape[0])}},
            {indices_W_2.data, {checked_cast(indices_W_2.shape[0])}},
            {indices_output_1.data, {checked_cast(indices_output_1.shape[0])}},
            {indices_output_2.data,
             {checked_cast(indices_output_2.shape[0])}}););
}

extern "C" int mops_sparse_accumulation_scatter_add_with_weights_f64(
    mops_tensor_3d_f64_t output, mops_tensor_2d_f64_t A, mops_tensor_2d_f64_t B,
    mops_tensor_1d_f64_t C, mops_tensor_3d_f64_t W,
    mops_tensor_1d_i32_t indices_A, mops_tensor_1d_i32_t indices_W_1,
    mops_tensor_1d_i32_t indices_W_2, mops_tensor_1d_i32_t indices_output_1,
    mops_tensor_1d_i32_t indices_output_2) {
    MOPS_CATCH_EXCEPTIONS(
        mops::sparse_accumulation_scatter_add_with_weights<double>(
            {output.data,
             {checked_cast(output.shape[0]), checked_cast(output.shape[1]),
              checked_cast(output.shape[2])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {W.data,
             {checked_cast(W.shape[0]), checked_cast(W.shape[1]),
              checked_cast(W.shape[2])}},
            {indices_A.data, {checked_cast(indices_A.shape[0])}},
            {indices_W_1.data, {checked_cast(indices_W_1.shape[0])}},
            {indices_W_2.data, {checked_cast(indices_W_2.shape[0])}},
            {indices_output_1.data, {checked_cast(indices_output_1.shape[0])}},
            {indices_output_2.data,
             {checked_cast(indices_output_2.shape[0])}}););
}

extern "C" int mops_sparse_accumulation_scatter_add_with_weights_vjp_f32(
    mops_tensor_2d_f32_t grad_A, mops_tensor_2d_f32_t grad_B,
    mops_tensor_3d_f32_t grad_W, mops_tensor_3d_f32_t grad_output,
    mops_tensor_2d_f32_t A, mops_tensor_2d_f32_t B, mops_tensor_1d_f32_t C,
    mops_tensor_3d_f32_t W, mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_W_1, mops_tensor_1d_i32_t indices_W_2,
    mops_tensor_1d_i32_t indices_output_1,
    mops_tensor_1d_i32_t indices_output_2) {
    MOPS_CATCH_EXCEPTIONS(
        mops::sparse_accumulation_scatter_add_with_weights_vjp<float>(
            {grad_A.data,
             {checked_cast(grad_A.shape[0]), checked_cast(grad_A.shape[1])}},
            {grad_B.data,
             {checked_cast(grad_B.shape[0]), checked_cast(grad_B.shape[1])}},
            {grad_W.data,
             {checked_cast(grad_W.shape[0]), checked_cast(grad_W.shape[1]),
              checked_cast(W.shape[2])}},
            {grad_output.data,
             {checked_cast(grad_output.shape[0]),
              checked_cast(grad_output.shape[1]),
              checked_cast(grad_output.shape[2])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {W.data,
             {checked_cast(W.shape[0]), checked_cast(W.shape[1]),
              checked_cast(W.shape[2])}},
            {indices_A.data, {checked_cast(indices_A.shape[0])}},
            {indices_W_1.data, {checked_cast(indices_W_1.shape[0])}},
            {indices_W_2.data, {checked_cast(indices_W_2.shape[0])}},
            {indices_output_1.data, {checked_cast(indices_output_1.shape[0])}},
            {indices_output_2.data,
             {checked_cast(indices_output_2.shape[0])}}););
}

extern "C" int mops_sparse_accumulation_scatter_add_with_weights_vjp_f64(
    mops_tensor_2d_f64_t grad_A, mops_tensor_2d_f64_t grad_B,
    mops_tensor_3d_f64_t grad_W, mops_tensor_3d_f64_t grad_output,
    mops_tensor_2d_f64_t A, mops_tensor_2d_f64_t B, mops_tensor_1d_f64_t C,
    mops_tensor_3d_f64_t W, mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_W_1, mops_tensor_1d_i32_t indices_W_2,
    mops_tensor_1d_i32_t indices_output_1,
    mops_tensor_1d_i32_t indices_output_2) {
    MOPS_CATCH_EXCEPTIONS(
        mops::sparse_accumulation_scatter_add_with_weights_vjp<double>(
            {grad_A.data,
             {checked_cast(grad_A.shape[0]), checked_cast(grad_A.shape[1])}},
            {grad_B.data,
             {checked_cast(grad_B.shape[0]), checked_cast(grad_B.shape[1])}},
            {grad_W.data,
             {checked_cast(grad_W.shape[0]), checked_cast(grad_W.shape[1]),
              checked_cast(W.shape[2])}},
            {grad_output.data,
             {checked_cast(grad_output.shape[0]),
              checked_cast(grad_output.shape[1]),
              checked_cast(grad_output.shape[2])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {W.data,
             {checked_cast(W.shape[0]), checked_cast(W.shape[1]),
              checked_cast(W.shape[2])}},
            {indices_A.data, {checked_cast(indices_A.shape[0])}},
            {indices_W_1.data, {checked_cast(indices_W_1.shape[0])}},
            {indices_W_2.data, {checked_cast(indices_W_2.shape[0])}},
            {indices_output_1.data, {checked_cast(indices_output_1.shape[0])}},
            {indices_output_2.data,
             {checked_cast(indices_output_2.shape[0])}}););
}

extern "C" int mops_sparse_accumulation_scatter_add_with_weights_vjp_f32(
    mops_tensor_2d_f32_t grad_A,
    mops_tensor_2d_f32_t grad_B,
    mops_tensor_3d_f32_t grad_W,
    mops_tensor_3d_f32_t grad_output,
    mops_tensor_2d_f32_t A,
    mops_tensor_2d_f32_t B,
    mops_tensor_1d_f32_t C,
    mops_tensor_3d_f32_t W,
    mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_W_1,
    mops_tensor_1d_i32_t indices_W_2,
    mops_tensor_1d_i32_t indices_output_1,
    mops_tensor_1d_i32_t indices_output_2
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::sparse_accumulation_scatter_add_with_weights_vjp<float>(
            {grad_A.data, {checked_cast(grad_A.shape[0]), checked_cast(grad_A.shape[1])}},
            {grad_B.data, {checked_cast(grad_B.shape[0]), checked_cast(grad_B.shape[1])}},
            {grad_W.data, {checked_cast(grad_W.shape[0]), checked_cast(grad_W.shape[1]), checked_cast(W.shape[2])}},
            {grad_output.data, {checked_cast(grad_output.shape[0]), checked_cast(grad_output.shape[1]), checked_cast(grad_output.shape[2])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {W.data, {checked_cast(W.shape[0]), checked_cast(W.shape[1]), checked_cast(W.shape[2])}},
            {indices_A.data, {checked_cast(indices_A.shape[0])}},
            {indices_W_1.data, {checked_cast(indices_W_1.shape[0])}},
            {indices_W_2.data, {checked_cast(indices_W_2.shape[0])}},
            {indices_output_1.data, {checked_cast(indices_output_1.shape[0])}},
            {indices_output_2.data, {checked_cast(indices_output_2.shape[0])}}
        );
    );
}


extern "C" int mops_sparse_accumulation_scatter_add_with_weights_vjp_f64(
    mops_tensor_2d_f64_t grad_A,
    mops_tensor_2d_f64_t grad_B,
    mops_tensor_3d_f64_t grad_W,
    mops_tensor_3d_f64_t grad_output,
    mops_tensor_2d_f64_t A,
    mops_tensor_2d_f64_t B,
    mops_tensor_1d_f64_t C,
    mops_tensor_3d_f64_t W,
    mops_tensor_1d_i32_t indices_A,
    mops_tensor_1d_i32_t indices_W_1,
    mops_tensor_1d_i32_t indices_W_2,
    mops_tensor_1d_i32_t indices_output_1,
    mops_tensor_1d_i32_t indices_output_2
) {
    MOPS_CATCH_EXCEPTIONS(
        mops::sparse_accumulation_scatter_add_with_weights_vjp<double>(
            {grad_A.data, {checked_cast(grad_A.shape[0]), checked_cast(grad_A.shape[1])}},
            {grad_B.data, {checked_cast(grad_B.shape[0]), checked_cast(grad_B.shape[1])}},
            {grad_W.data, {checked_cast(grad_W.shape[0]), checked_cast(grad_W.shape[1]), checked_cast(W.shape[2])}},
            {grad_output.data, {checked_cast(grad_output.shape[0]), checked_cast(grad_output.shape[1]), checked_cast(grad_output.shape[2])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {W.data, {checked_cast(W.shape[0]), checked_cast(W.shape[1]), checked_cast(W.shape[2])}},
            {indices_A.data, {checked_cast(indices_A.shape[0])}},
            {indices_W_1.data, {checked_cast(indices_W_1.shape[0])}},
            {indices_W_2.data, {checked_cast(indices_W_2.shape[0])}},
            {indices_output_1.data, {checked_cast(indices_output_1.shape[0])}},
            {indices_output_2.data, {checked_cast(indices_output_2.shape[0])}}
        );
    );
}


extern "C" int mops_cuda_sparse_accumulation_scatter_add_with_weights_f32(
    mops_tensor_3d_f32_t output, mops_tensor_2d_f32_t A, mops_tensor_2d_f32_t B,
    mops_tensor_1d_f32_t C, mops_tensor_3d_f32_t W,
    mops_tensor_1d_i32_t indices_A, mops_tensor_1d_i32_t indices_W_1,
    mops_tensor_1d_i32_t indices_W_2, mops_tensor_1d_i32_t indices_output_1,
    mops_tensor_1d_i32_t indices_output_2) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::sparse_accumulation_scatter_add_with_weights<float>(
            {output.data,
             {checked_cast(output.shape[0]), checked_cast(output.shape[1]),
              checked_cast(output.shape[2])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {W.data,
             {checked_cast(W.shape[0]), checked_cast(W.shape[1]),
              checked_cast(W.shape[2])}},
            {indices_A.data, {checked_cast(indices_A.shape[0])}},
            {indices_W_1.data, {checked_cast(indices_W_1.shape[0])}},
            {indices_W_2.data, {checked_cast(indices_W_2.shape[0])}},
            {indices_output_1.data, {checked_cast(indices_output_1.shape[0])}},
            {indices_output_2.data,
             {checked_cast(indices_output_2.shape[0])}}););
}

extern "C" int mops_cuda_sparse_accumulation_scatter_add_with_weights_f64(
    mops_tensor_3d_f64_t output, mops_tensor_2d_f64_t A, mops_tensor_2d_f64_t B,
    mops_tensor_1d_f64_t C, mops_tensor_3d_f64_t W,
    mops_tensor_1d_i32_t indices_A, mops_tensor_1d_i32_t indices_W_1,
    mops_tensor_1d_i32_t indices_W_2, mops_tensor_1d_i32_t indices_output_1,
    mops_tensor_1d_i32_t indices_output_2) {
    MOPS_CATCH_EXCEPTIONS(
        mops::cuda::sparse_accumulation_scatter_add_with_weights<double>(
            {output.data,
             {checked_cast(output.shape[0]), checked_cast(output.shape[1]),
              checked_cast(output.shape[2])}},
            {A.data, {checked_cast(A.shape[0]), checked_cast(A.shape[1])}},
            {B.data, {checked_cast(B.shape[0]), checked_cast(B.shape[1])}},
            {C.data, {checked_cast(C.shape[0])}},
            {W.data,
             {checked_cast(W.shape[0]), checked_cast(W.shape[1]),
              checked_cast(W.shape[2])}},
            {indices_A.data, {checked_cast(indices_A.shape[0])}},
            {indices_W_1.data, {checked_cast(indices_W_1.shape[0])}},
            {indices_W_2.data, {checked_cast(indices_W_2.shape[0])}},
            {indices_output_1.data, {checked_cast(indices_output_1.shape[0])}},
            {indices_output_2.data,
             {checked_cast(indices_output_2.shape[0])}}););
}
