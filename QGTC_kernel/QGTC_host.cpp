#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#define min(x, y) (((x) < (y))? (x) : (y))

// a two layer GraphSage
// std::vector<torch::Tensor> QGTC_forward_cuda(
//     torch::Tensor A_mat,
//     torch::Tensor X_mat,
//     torch::Tensor W_1,
//     torch::Tensor W_2,
//     const int w_bit,
//     const int act_bit
// );

// GPU kernel for quantization,
// bit decomposition, and accumulation
torch::Tensor bit_qnt_cuda(
    torch::Tensor input,
    const int bit_qnt,
    const bool col_major=false
);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//////////////////////////////////////////
//
// QGTC forward function for AX = X_hat 
// neighbor aggregation of GNN.
// A can be 1-bit compressed.
// X_mat is M-bit with 1-bit compression in each layer.
// 
////////////////////////////////////////////
// std::vector<torch::Tensor> QGTC_forward(
//     torch::Tensor A_mat,
//     torch::Tensor X_mat,
//     torch::Tensor W_1,
//     torch::Tensor W_2,
//     const int w_bit,
//     const int act_bit,
// ) {
//   CHECK_INPUT(A_mat);
//   CHECK_INPUT(X_mat);
//   CHECK_INPUT(W_1);
//   CHECK_INPUT(W_2);

//   return QGTC_forward_cuda(A_mat, X_mat, W_1, W_2, w_bit, act_bit);
// }

torch::Tensor bit_qnt(
    torch::Tensor input,
    const int bit_qnt,
    const bool col_major=false
){
  CHECK_INPUT(input);
  return bit_qnt_cuda(input, bit_qnt, col_major);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bit_qnt", &bit_qnt, "quantize a float --> bit tensor (CUDA)");
  // m.def("forward", &QGTC_forward, "QGTC AX = X_hat forward (CUDA)");
}