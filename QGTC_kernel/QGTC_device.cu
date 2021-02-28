#include <torch/extension.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>

#include "config.h"
#define WPB 8

using namespace nvcuda;

//////////////////////
/// SPMM forward (GCN, GraphSAGE)
//////////////////////
__global__ void QGTC_forward_cuda_kernel(
    torch::Tensor A_mat,
    torch::Tensor X_mat
);

////////////////////////////////////////////
//
// SPMM Foward Pass  (GCN, GraphSAGE)
//
////////////////////////////////////////////
std::vector<torch::Tensor> QGTC_forward_cuda(
    torch::Tensor A_mat,
    torch::Tensor X_mat,
    const int w_bit,
    const int act_bit
) 
{
    auto output = torch::zeros_like(input);

    // quant_A from float to 1-w_bit.
    // quant X from float to act_bit.
    // quant W_1 from float to w_bit.
    // quant W_2 from float to w_bit.

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
}