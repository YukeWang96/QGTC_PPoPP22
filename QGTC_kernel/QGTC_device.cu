#include <torch/extension.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>

#include "utility.h"
#include "kernel.h"

using namespace nvcuda;

//
// quantize the input float --> uint32 1-bit
//
torch::Tensor bit_qnt_cuda(
    torch::Tensor input,
    const int bit_qnt,
    const bool col_major=false,
    const bool output_layer=false
){
    const int height = input.size(0);
    const int width = input.size(1);

    const int dev = 0;
    const int numThreads = 1024;
    int numBlocksPerSm;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Quantize_val, numThreads, 0);
    
    // quantization float --> int32
    // note that allocated data must be on CUDA device !!!!
    torch::Tensor input_qnt = torch::zeros({height, width}, torch::kInt32).to(torch::kCUDA);
    // printf("-- Input_qnt\n");

    Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(input_qnt.data<int>(), input.data<float>(), 
                                                                                height*width, bit_qnt); 

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error at Quantize_val: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // column-major store for weight compression.
    if (col_major)
    {
        // printf("==> column major\n");
        // allocate output in uint32.
        if (output_layer){
            auto output = torch::zeros({bit_qnt*STEP32(height), PAD8(width)}, torch::kInt32).to(torch::kCUDA);         // PAD(8) -- output

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128_OUTPUT, numThreads, 0);
            PackFcWeight128_OUTPUT<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
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
        else{
            auto output = torch::zeros({bit_qnt*STEP32(height), PAD128(width)}, torch::kInt32).to(torch::kCUDA);         // PAD(128) -- hidden

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
    }
    else // row-major store for input compression.
    {
        // printf("==> Non-column major\n");
        // allocate output in int32 on GPU
        torch::Tensor output = torch::zeros({bit_qnt*PAD8(height), STEP32(width)}, torch::kInt32).to(torch::kCUDA);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_input, numThreads, 0);

        // printf("bit_qnt: %d, height: %d, width: %d\n", bit_qnt, height, width);
        // printf("numThreads: %d, numBlocksPerSm: %d\n", numThreads, numBlocksPerSm);

        // on the second iterations fails when qnt_bit=1.
        QGTC_layer_input<<< numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>> \
                (output.data<int>(), input_qnt.data<int>(), height, width, bit_qnt);
   
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error at mm_v1_cuda: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        return output;
    }
}

//
// bit_X1 and bit_x2 --> [ int32 ] output.
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
    // allocate the output Tensor on GPU.
    auto bit_X_out = torch::zeros({output_bit*X1_height, STEP32(PAD128(X2_width))}, torch::kInt32).to(torch::kCUDA);
    
    int dev = 0;
    int numThreads = 512;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    // int shared_memory = 64*sizeof(int)*32;
    int shared_memory = 256*sizeof(int)*32;

    cudaGetDeviceProperties(&deviceProp, dev);
    cudaFuncSetAttribute(QGTC_layer_hidden, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_hidden, numThreads, shared_memory);

    QGTC_layer_hidden<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads, shared_memory>>>(
        bit_X_out.data<int>(), bit_X1.data<int>(), bit_X2.data<int>(),
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
// bit_X1 and bit_x2 --> [ float ] output.
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
    // allocate the output Tensor on GPU.
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    // printf("X1_height: %d, X2_width: %d\n", X1_height, X2_width);

    torch::Tensor float_X_out = torch::zeros({X1_height, PAD8(X2_width)}).to(torch::kCUDA);
    // int out_height = float_X_out.size(0);
    // int out_width = float_X_out.size(1);
    // printf("out_height: %d\n", out_height);
    // printf("out_height: %d, out_width: %d\n", out_height, out_width);
    // exit(-1);

    int dev = 0;
    int numThreads = 128;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int shared_memory = 64*sizeof(int)*32;

    cudaGetDeviceProperties(&deviceProp, dev);
    cudaFuncSetAttribute(QGTC_layer_output, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_output, numThreads, shared_memory);

    // printf("QGTC_layer_output\n");
    QGTC_layer_output<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads, shared_memory>>>(
        float_X_out.data<float>(), bit_X1.data<int>(), bit_X2.data<int>(),
        X1_height, X1_width, X2_width, bit1, bit2
    );

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error at mm_v2_cuda: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return float_X_out;
}