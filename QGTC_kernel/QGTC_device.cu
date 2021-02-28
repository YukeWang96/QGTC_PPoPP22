#include <torch/extension.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>

#include "config.h"
#include "utility.h"
#include "kernel.cuh"

using namespace nvcuda;

//
// input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
//
__global__ void Quantize_val(
    uint32_t* input_qnt_gpu, 
    float* __restrict__ input_gpu, 
    const int num_elements, 
    const int bitwidth)
{
    int start = blockIdx.x * blockDim.x + threadIdx.x;

    for (int tid = start; tid < num_elements; tid += blockDim.x*gridDim.x) {
        /*
        * Quant_val  - 0            2^{bitwidth}    
        *-------------------- = ------------------
        * Actual_val - min_val  max_val - min_val
        */
        float input_val = clip(input_gpu[tid], min_v, max_v);
        float qnt_float = (input_val - min_v) * (1 << bitwidth) * 1.0f / (max_v - min_v);
        input_qnt_gpu[tid]  =  __float2uint_rn(qnt_float);
    }
}  

//
// 
//
torch::Tensor bit_qnt_cuda(
    torch::Tensor input,
    const int bit_qnt,
    const bool col_major=false
){
    const int height = input.size(0);
    const int width = input.size(1);

    const int dev = 0;
    const int numThreads = 1024;
    int numBlocksPerSm;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Quantize_val,  numThreads, 0);
    
    // quantization float --> uint32
    auto input_qnt = torch::zeros((height, width));
    Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>> \
                                    (input_qnt.data<uint32_t>(), input.data<float>(), height*width, bit_qnt); 

    // column-major store for weight compression.
    if (col_major)
    {
        auto output = torch::zeros((bit_qnt*STEP32(height), width));

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128, numThreads, 0);
        PackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            output.data<uint32_t>(), input_qnt.data<uint32_t>(),
            height, width, bit_qnt
        );
        return output;
    }
    else // row-major store for input compression.
    {
        auto output = torch::zeros((bit_qnt*height, STEP32(width)));

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_input,  numThreads, 0);
        QGTC_layer_input<<< numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>> \
                (output.data<uint32_t>(), input_qnt.data<uint32_t>(),
                height, width, bit_qnt);
                
        return output;
    }
}

//
// bit_X1 and bit_x2 --> float output.
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
)
{
    auto bit_X_out = torch::zeros((output_bit*X1_height, STEP32(X2_width)));
    
    int dev = 0;
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int shared_memory = 64*sizeof(int)*32;

    cudaGetDeviceProperties(&deviceProp, dev);
    cudaFuncSetAttribute(QGTC_layer_hidden, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_hidden, numThreads, shared_memory);

    QGTC_layer_hidden<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads, shared_memory>>>(
        bit_X_out.data<uint32_t>(), bit_X1.data<uint32_t>(), bit_X2.data<uint32_t>(),
        X1_height, X1_width, X2_width, bit1, bit2
    );

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        // print the CUDA error message and exit
        printf("CUDA error at mm_v1_cuda: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return bit_X_out;
}


//
// bit_X1 and bit_x2 --> bit output.
//
torch::Tensor mm_v2_cuda(
    torch::Tensor bit_X1,
    torch::Tensor bit_X2,
    const int X1_height,
    const int X1_width,
    const int X2_width,
    const int bit1,
    const int bit2
)
{
    auto float_X_out = torch::zeros((X1_height, X2_width));
    
    int dev = 0;
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int shared_memory = 64*sizeof(int)*32;

    cudaGetDeviceProperties(&deviceProp, dev);
    cudaFuncSetAttribute(QGTC_layer_hidden, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_hidden, numThreads, shared_memory);

    QGTC_layer_output<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads, shared_memory>>>(
        float_X_out.data<float>(), bit_X1.data<uint32_t>(), bit_X2.data<uint32_t>(),
        X1_height, X1_width, X2_width, bit1, bit2
    );

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error at mm_v2_cuda: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return float_X_out;
}
