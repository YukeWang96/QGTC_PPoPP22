#include <torch/extension.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>

#include "utility.h"
#include "kernel.h"

using namespace nvcuda;

__global__ void test_kernel(int * __restrict__ input){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    printf("%d \n", idx);
}

//
// quantize the input float --> uint32 1-bit
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
    torch::Tensor input_qnt = torch::zeros({height, width}, torch::kInt32);
    printf("input_qnt\n");

    // test_kernel<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(input_qnt.data<int>());
    Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(input_qnt.data<int>(), input.data<float>(), 
                                                                                height*width, bit_qnt); 

    printf("Quantize_val\n");

    // column-major store for weight compression.
    if (col_major)
    {
        // allocate output in uint32.
        auto output = torch::zeros({bit_qnt*STEP32(height), PAD8(width)}, torch::kInt32);

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128, numThreads, 0);
        PackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            output.data<int>(), input_qnt.data<int>(),
            height, width, bit_qnt
        );

        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error at mm_v1_cuda: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        return output;
    }
    else // row-major store for input compression.
    {

        printf("non column major\n");

        // allocate output in uint32.
        auto output = torch::zeros({bit_qnt*PAD8(height), STEP32(width)}, torch::kInt32);
        // auto output = torch::zeros((2, 3));

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_input, numThreads, 0);
        QGTC_layer_input<<< numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>> \
                (output.data<int>(), input_qnt.data<int>(),
                height, width, bit_qnt);

        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error at mm_v1_cuda: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        return output;
    }
}

/*
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
*/