#include <torch/extension.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>

#include "utility.h"
#include "kernel.h"

using namespace nvcuda;

void print_IntTensor_cpu(torch::Tensor input){
    int height = input.size(0);
    int width = input.size(1);

    auto input_acc = input.accessor<int, 2>();
    for (int h = 0; h < height; h++){
        for (int w = 0; w < width; w++){
            printf("%d ", input_acc[h][w]);
        }
        printf("\n");
    }
    printf("------------------------------\n");
}

__global__
void print_IntTensor_cuda(torch::PackedTensorAccessor64<int, 2> input, int height, int width){

    for (int h = 0; h < height; h++){
        for (int w = 0; w < width; w++){
            printf("%d ", input[h][w]);
        }
        printf("\n");
    }
    printf("------------------------------\n");
}


//
// 1. Encoding float --> uint32 1-bit
//
torch::Tensor val2bit_cuda(
    torch::Tensor input_val,
    const int nbits,
    const bool col_major=false,
    const bool output_layer=false
){
    const int height = input_val.size(0);
    const int width = input_val.size(1);

    const int dev = 0;
    int numBlocksPerSm;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, Quantize_val, numThreads, 0);

    // printf("Before quantize value\n");
    // quantization float --> int32
    auto input_qnt = torch::zeros({height, width}, torch::kInt32).to(torch::kCUDA);
    Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(input_qnt.data<int>(), 
                                                                                input_val.data<float>(), 
                                                                                height*width, nbits); 
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error at Quantize_val: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // printf("Before print_IntTensor\n");
    // print_IntTensor(input_qnt);
    // print_IntTensor_cuda<<<1,1>>>(input_qnt.packed_accessor64<int,2>(), input_qnt.size(0), input_qnt.size(1));

    // Column-major bit-decoding Weight.
    // Hidden and output layer compression seperately.
    if (col_major)
    {
        // printf("==> bit2val_cuda::column major\n");
        if (output_layer){
            auto output_bit= torch::zeros({nbits*STEP128(height)*4, PAD8(width)}, torch::kInt32).to(torch::kCUDA);

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128, numThreads, 0);
            PackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                output_bit.data<int>(), input_qnt.data<int>(), height, width, nbits);

            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess){
                printf("CUDA error at bit2val_cuda::column_major::output_layer: %s\n", cudaGetErrorString(error));
                exit(-1);
            }
            return output_bit;
        }
        else{ // column-major hidden layer. for W and XW.
            auto output_bit = torch::zeros({nbits*STEP128(height)*4, PAD128(width)}, torch::kInt32).to(torch::kCUDA);
                                           // {nbits*PAD8(height), STEP128(width)*4}

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, PackFcWeight128, numThreads, 0);
            PackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                output_bit.data<int>(), input_qnt.data<int>(), height, width, nbits);

            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess){
                printf("CUDA error at bit2val_cuda::column_major::hiddn: %s\n", cudaGetErrorString(error));
                exit(-1);
            }
            return output_bit;
        }
    }
    else // row-major bit-decoding activation
    {
        // printf("==> bit2val_cuda::row_major\n");
        auto output_bit = torch::zeros({nbits*PAD8(height), STEP128(width)*4}, torch::kInt32).to(torch::kCUDA);
        // printf("output_bit shape: %d, %d\n", output_bit.size(0), output_bit.size(1));
        // printf("input_qnt shape: %d, %d\n", height, width);

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_input, numThreads, 0);
        QGTC_layer_input<<< numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>> \
                (output_bit.data<int>(), input_qnt.data<int>(), height, width, nbits);
   
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error at bitMM2Bit_cuda: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        return output_bit;
    }
}

//
// 2. Decoding the compressed bit (uint32) --> val (uint32).
//
torch::Tensor bit2val_cuda(
    torch::Tensor input_bit,
    const int nbits,
    const int height,
    const int width,
    const bool col_major=false,
    const bool output_layer=false)
{
 
    const int dev = 0;
    int numBlocksPerSm;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    // column-major store for [ weight ] decoding.
    if (col_major)
    {
        if (output_layer){
            auto output_val = torch::zeros({height, width}, torch::kInt32).to(torch::kCUDA);         // PAD(8) -- output
            
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcWeight128, 
                                                        numThreads, 0);
            UnPackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>\
                                    (output_val.data<int>(), input_bit.data<int>(), height, width, nbits);
           
            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess){
                printf("CUDA error at bitMM2Bit_cuda::col_major::output_layer, : %s\n", cudaGetErrorString(error));
                exit(-1);
            }
            return output_val;
        }
        else{ // column-major hidden layer
            auto output_val = torch::zeros({height, width}, torch::kInt32).to(torch::kCUDA);         // PAD(128) -- hidden
            printf("output_val shape: %d, %d\n", output_val.size(0), output_val.size(1));

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcWeight128, 
                                                            numThreads, 0);
            UnPackFcWeight128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
                    output_val.data<int>(), input_bit.data<int>(), height, width, nbits);

            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess){
                printf("CUDA error at bitMM2Bit_cuda: %s\n", cudaGetErrorString(error));
                exit(-1);
            }
            return output_val;
        }

    }
    else // Row-major bit-decoding for Activation. 
    {
        printf("==> Row-major bit-decoding \n");
        auto output_val = torch::zeros({height, width}, torch::kInt32).to(torch::kCUDA);

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, UnPackFcOutput128, 
                                                        numThreads, 0);
        UnPackFcOutput128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>>(
            output_val.data<int>(), input_bit.data<int>(), height, width, nbits);

        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            printf("CUDA error at bitMM2Bit_cuda: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        printf("output_val shape: %d, %d\n", output_val.size(0), output_val.size(1));
        print_IntTensor_cuda<<<1,1>>>(output_val.packed_accessor64<int,2>(), output_val.size(0), output_val.size(1));

        return output_val;
    }
}

//
// bit_X1 and bit_x2 --> [ int32 ] in bit output.
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
)
{
    // allocate the output Tensor on GPU.
    auto bit_X_out = torch::zeros({output_bit*PAD8(X1_height), STEP128(X2_width)*4}, torch::kInt32).to(torch::kCUDA);
    
    int dev = 0;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int shared_memory = 64*1e3; // 64KB

    cudaGetDeviceProperties(&deviceProp, dev);
    cudaFuncSetAttribute(QGTC_layer_hidden, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_hidden, numThreads_1, shared_memory);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    #define PROF 1
    for (int i = 0; i < PROF; i++)
    QGTC_layer_hidden<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads_1, shared_memory>>>(
        bit_X_out.data<int>(), bit_X1.data<int>(), bit_X2.data<int>(),
        X1_height, X1_width, X2_width, bit1, bit2, output_bit);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // printf("X1_height %d, X1_width: %d, X2_width: %d, TFLOPs: %.3f\n", \
    //     X1_height, X1_width, X2_width, \
    //     2.0f*X1_height*X1_width*X2_width*PROF/(milliseconds/1e3)/1e12);


    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error at bitMM2Bit_cuda: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    // print_counter<<<1,1>>>();
    // printf("counter: %d\n", counter);
    return bit_X_out;
}


//
// bit_X1 and bit_x2 --> [ int32 ] in bit output.
//
torch::Tensor bitMM2Bit_cuda_base_cnt(
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
    auto bit_X_out = torch::zeros({output_bit*PAD8(X1_height), STEP128(X2_width)*4}, torch::kInt32).to(torch::kCUDA);
    
    int dev = 0;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int shared_memory = 64*1e3; // 64KB

    cudaGetDeviceProperties(&deviceProp, dev);
    cudaFuncSetAttribute(QGTC_layer_hidden_base_cnt, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_hidden_base_cnt, numThreads_1, shared_memory);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    QGTC_layer_hidden_base_cnt<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads_1, shared_memory>>>(
        bit_X_out.data<int>(), bit_X1.data<int>(), bit_X2.data<int>(),
        X1_height, X1_width, X2_width, bit1, bit2, output_bit);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error at bitMM2Bit_cuda_base_cnt: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    print_counter_global<<<1,1>>>();
    return bit_X_out;
}



//
// bit_X1 and bit_x2 --> [ int32 ] in bit output.
//
torch::Tensor bitMM2Bit_cuda_zerojump_cnt(
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
    auto bit_X_out = torch::zeros({output_bit*PAD8(X1_height), STEP128(X2_width)*4}, torch::kInt32).to(torch::kCUDA);
    
    int dev = 0;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int shared_memory = 64*1e3; // 64KB

    cudaGetDeviceProperties(&deviceProp, dev);
    cudaFuncSetAttribute(QGTC_layer_hidden_zerojump_cnt, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_hidden_zerojump_cnt, numThreads_1, shared_memory);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    QGTC_layer_hidden_zerojump_cnt<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads_1, shared_memory>>>(
        bit_X_out.data<int>(), bit_X1.data<int>(), bit_X2.data<int>(),
        X1_height, X1_width, X2_width, bit1, bit2, output_bit);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error at QGTC_layer_hidden_zerojump_cnt: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    print_counter<<<1,1>>>();
    return bit_X_out;
}


//
// bit_X1 and bit_x2 --> [ int32 ] in bit output. profiling for 200 rounds.
//
torch::Tensor bitMM2Bit_cuda_profile(
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
    auto bit_X_out = torch::zeros({output_bit*PAD8(X1_height), STEP128(X2_width)*4}, torch::kInt32).to(torch::kCUDA);
    
    int dev = 0;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int shared_memory = 64*1e3; // 64KB

    cudaGetDeviceProperties(&deviceProp, dev);
    cudaFuncSetAttribute(QGTC_layer_hidden, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_hidden, numThreads_1, shared_memory);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    #define PROF 200
    for (int i = 0; i < PROF; i++)
        QGTC_layer_hidden<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads_1, shared_memory>>>(
            bit_X_out.data<int>(), bit_X1.data<int>(), bit_X2.data<int>(),
            X1_height, X1_width, X2_width, bit1, bit2, output_bit);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("X1_height %d, X1_width: %d, X2_width: %d, TFLOPs: %.3f\n", \
        X1_height, X1_width, X2_width, \
        2.0f*X1_height*X1_width*X2_width*PROF/(milliseconds/1e3)/1e12);


    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error at bitMM2Bit_cuda: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    // print_counter<<<1,1>>>();
    // printf("counter: %d\n", counter);
    return bit_X_out;
}



//
// bit_X1 and bit_x2 --> [ int32 ] output.
//
torch::Tensor bitMM2Bit_col_cuda(
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

    //refer to Weights padding strategy: nbits*STEP128(height)*4, PAD128(width)
    // allocate the output Tensor on GPU.
    // auto bit_X_out = torch::zeros({output_bit*PAD8(X1_height), STEP128(X2_width)*4}, torch::kInt32).to(torch::kCUDA);
    auto bit_X_out = torch::zeros({output_bit*STEP128(X1_height)*4, PAD128(X2_width)}, torch::kInt32).to(torch::kCUDA);

    int dev = 0;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int shared_memory = 32*1e3; // 64KB

    cudaGetDeviceProperties(&deviceProp, dev);
    cudaFuncSetAttribute(QGTC_layer_hidden_col, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_hidden_col, numThreads_1, shared_memory);
    printf("numBlocksPerSm*deviceProp.multiProcessorCount: %d, numThreads_1: %d, shared_memory: %d\n", \
    numBlocksPerSm*deviceProp.multiProcessorCount, numThreads_1, shared_memory);

    QGTC_layer_hidden_col<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads_1, shared_memory>>>(
        bit_X_out.data<int>(), bit_X1.data<int>(), bit_X2.data<int>(),
        X1_height, X1_width, X2_width, bit1, bit2, output_bit);

    // cudaFuncSetAttribute(QGTC_layer_hidden, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_hidden, numThreads_1, shared_memory);

    // QGTC_layer_hidden<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads_1, shared_memory>>>(
    //     bit_X_out.data<int>(), bit_X1.data<int>(), bit_X2.data<int>(),
    //     X1_height, X1_width, X2_width, bit1, bit2, output_bit);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error at bitMM2Bit_col_cuda: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    // print_counter<<<1,1>>>();
    // printf("counter: %d\n", counter);
    return bit_X_out;
}


//
// bit_X1 and bit_x2 --> [ float ] output.
//
torch::Tensor bitMM2Int_cuda(
    torch::Tensor bit_X1,
    torch::Tensor bit_X2,
    const int X1_height,
    const int X1_width,
    const int X2_width,
    const int bit1,
    const int bit2,
    const int pad_128=false
)
{
    // allocate the output Tensor on GPU.
    torch::Tensor float_X_out = torch::zeros({X1_height, X2_width}).to(torch::kCUDA);

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    int numBlocksPerSm;
    int shared_memory = 64*1e3; // 64KB

    if (pad_128){   // for GIN like A (XW), XW is PAD128
        cudaFuncSetAttribute(QGTC_layer_output_PAD128, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_output_PAD128, numThreads_1, shared_memory);
    
        QGTC_layer_output_PAD128<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads_1, shared_memory>>>(
            float_X_out.data<float>(), bit_X1.data<int>(), bit_X2.data<int>(),
            X1_height, X1_width, X2_width, bit1, bit2
        );
    }
    else{   // for all other case C = XW, W is PAD8.
        cudaFuncSetAttribute(QGTC_layer_output_PAD8, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_output_PAD8, numThreads_1, shared_memory);
    
        QGTC_layer_output_PAD8<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads_1, shared_memory>>>(
            float_X_out.data<float>(), bit_X1.data<int>(), bit_X2.data<int>(),
            X1_height, X1_width, X2_width, bit1, bit2
        );
    }

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error at bitMM2Int_cuda: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return float_X_out;
}