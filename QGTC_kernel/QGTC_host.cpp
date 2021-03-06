#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#define min(x, y) (((x) < (y))? (x) : (y))

//
// bit_X1 and bit_x2 --> bit output.
//
torch::Tensor mm_v1_cuda(
    torch::Tensor bit_X1,
    torch::Tensor bit_X2,
    const int X1_height,
    const int X1_width,
    const int X2_width,
    const int bit1,
    const int bit2,
    const int output_bit
);

//
// bit_X1 and bit_x2 --> float output.
//
torch::Tensor mm_v2_cuda(
    torch::Tensor bit_X1,
    torch::Tensor bit_X2,
    const int X1_height,
    const int X1_width,
    const int X2_width,
    const int bit1,
    const int bit2
);

// GPU kernel for quantization
// and bit decomposition
torch::Tensor bit_qnt_cuda(
    torch::Tensor input,
    const int bit_qnt,
    const bool col_major=false
);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//
// bit_X1 and bit_x2 --> [ bit ] output.
//
torch::Tensor mm_v1(
    torch::Tensor bit_X1,
    torch::Tensor bit_X2,
    const int X1_height,
    const int X1_width,
    const int X2_width,
    const int bit1,
    const int bit2,
    const int output_bit
) {
  CHECK_INPUT(bit_X1);
  CHECK_INPUT(bit_X2);

  return mm_v1_cuda(bit_X1, bit_X2, X1_height, X1_width, X2_width, bit1, bit2, output_bit);
}

//
// bit_X1 and bit_X2 --> [ float ] output.
//
torch::Tensor mm_v2(
    torch::Tensor bit_X1,
    torch::Tensor bit_X2,
    const int X1_height,
    const int X1_width,
    const int X2_width,
    const int bit1,
    const int bit2
) {
  CHECK_INPUT(bit_X1);
  CHECK_INPUT(bit_X2);

  return mm_v2_cuda(bit_X1, bit_X2, X1_height, X1_width, X2_width, bit1, bit2);
}

//
// float input --> bit_input.
// e.g., [M x N] --> (ptr->shape)[bit x M x N]
//
torch::Tensor bit_qnt(
    torch::Tensor input,
    const int bit_qnt,
    const bool col_major=false
){
  CHECK_INPUT(input);
  
  // the pointer of the compressed address.
  return bit_qnt_cuda(input, bit_qnt, col_major); 
}

// binding to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bit_qnt", &bit_qnt, "quantize a [ float --> bit ] tensor (CUDA)");
  m.def("mm_v1", &mm_v1, "QGTC [ bit_X1 x bit_X2 --> bit_output ] forward (CUDA)");
  m.def("mm_v2", &mm_v2, "QGTC [ bit_X1 x bit_X2 --> float_output ] forward (CUDA)");
}