#include <torch/extension.h>

torch::Tensor val2bit_cuda(
    torch::Tensor input,
    const int nbits,
    const bool col_major=false,
    const bool output_layer=false // True: PAD8(width), False: PAD128(width)
);

torch::Tensor bit2val_cuda(
    torch::Tensor input,
    const int nbits,
    const int height,
    const int width,
    const bool col_major=false,
    const bool output_layer=false
);

//
// bit_X1 and bit_x2 --> bit output.
//
torch::Tensor bitMM2Bit_cuda(
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
torch::Tensor bitMM2Int_cuda(
    torch::Tensor bit_X1,
    torch::Tensor bit_X2,
    const int X1_height,
    const int X1_width,
    const int X2_width,
    const int bit1,
    const int bit2
);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//
// bit_X1 and bit_x2 --> [ bit ] output.
//
torch::Tensor bitMM2Bit(
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

  return bitMM2Bit_cuda(bit_X1, bit_X2, X1_height, X1_width, X2_width, bit1, bit2, output_bit);
}

//
// bit_X1 and bit_X2 --> [ float ] output.
//
torch::Tensor bitMM2Int(
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

  return bitMM2Int_cuda(bit_X1, bit_X2, X1_height, X1_width, X2_width, bit1, bit2);
}

//
// Bit-encoding: float input --> bit_input.
// int32 Tensor (M x N) --> int32 Tensor (bit x M x N/32)
//
torch::Tensor val2bit(
    torch::Tensor input,
    const int nbits,
    const bool col_major=false,
    const bool output_layer=false
){
  CHECK_INPUT(input);
  
  return val2bit_cuda(input, nbits, col_major, output_layer); 
}

//
// Bit-decoding: bit --> int32.
// int32 Tensor (nbits x M x N/32) --> int32 Tensor (M x N)
//
torch::Tensor bit2val(
    torch::Tensor input,
    const int nbits,
    const int height,
    const int width,
    const bool col_major=false,
    const bool output_layer=false
){
  CHECK_INPUT(input);

  return bit2val_cuda(input, nbits, height, width, col_major, output_layer); 
}

// Pytorch Binding.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("val2bit", &val2bit, "quantize a [ int32 --> bit ] tensor (CUDA)");
  m.def("bit2val", &bit2val, "quantize a [ bit --> int32 ] tensor (CUDA)");

  m.def("bitMM2Bit", &bitMM2Bit, "QGTC [ bit_A x bit_B --> bit_C ] forward (CUDA)");
  m.def("bitMM2Int", &bitMM2Int, "QGTC [ bit_A x bit_B --> int32_C ] forward (CUDA)");
}