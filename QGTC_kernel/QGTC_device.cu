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

//////////////////////
/// SPMM forward (GCN, GraphSAGE)
//////////////////////
// __global__ void QGTC_forward_cuda_kernel(
//     torch::Tensor A_mat,
//     torch::Tensor X_mat
// );


//////////////////////
/// float tensor --> uint32_t tensor.
//////////////////////
__global__ void Quantize_val(
    uint32_t* input_qnt_gpu, 
    float* __restrict__ input_gpu, 
    const int num_elements, 
    const int bitwidth
);

////////////////////////////////////////////
// quantize the input data into bit map.
////////////////////////////////////////////
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
    
    // quantization
    auto input_qnt = torch::zeros((height, width));
    Quantize_val<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>> \
                                    (input_qnt.data<uint32_t>(), input.data<float>(), height*width, bit_qnt); 

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
    else
    {
        auto output = torch::zeros((bit_qnt*height, STEP32(width)));

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, QGTC_layer_input,  numThreads, 0);
        QGTC_layer_input<<<numBlocksPerSm*deviceProp.multiProcessorCount, numThreads>>> \
                (output.data<uint32_t>(), input_qnt.data<uint32_t>(),
                height, width, bit_qnt);
                
        return output;
    }
}



// input_gpu (float 32-bit) -->  input_qnt_gpu (uint 32-bit)
/////////////////////////////////////////////////////
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



////////////////////////////////////////////
//
// SPMM Foward Pass  (GCN, GraphSAGE)
//
////////////////////////////////////////////
/*
std::vector<torch::Tensor> QGTC_forward_cuda(
    torch::Tensor A_mat,
    torch::Tensor X_mat,
    const int w_bit,
    const int act_bit
) 
{
    // auto output = torch::zeros_like(input);
    
    // dim3 grid(num_row_windows, 1, 1);
    // dim3 block(WARP_SIZE, WARPperBlock, 1);

    // const int dimTileNum = (embedding_dim + BLK_H - 1) / BLK_H;
	// const int dynamic_shared_size = dimTileNum*BLK_W * BLK_H * sizeof(float); // dynamic shared memory.





    // spmm_forward_cuda_kernel<<<grid, block, dynamic_shared_size>>>(
    //                                                                 nodePointer.data<int>(), 
    //                                                                 edgeList.data<int>(),
    //                                                                 blockPartition.data<int>(), 
    //                                                                 edgeToColumn.data<int>(), 
    //                                                                 edgeToRow.data<int>(), 
    //                                                                 num_nodes,
    //                                                                 num_edges,
    //                                                                 embedding_dim,
    //                                                                 input.data<float>(), 
    //                                                                 output.data<float>()
    //                                                             );

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return {output};
}*/