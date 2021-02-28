#ifndef COMPUTE_H
#define COMPUTE_H

#include "config.h"
#include "utility.h"

#define max_v 10
#define min_v -10


__inline__ __device__ float clip(float x, float lb, float ub){
    if (x < lb) return lb+1;
    if (x > ub) return ub-1;
    return x;
}

// * quantization of a single float value
__device__ __inline__ uin32 quantize(float val, int bitwidth){
    const int max_val = 10000;
    const int min_val = -max_val;
    if (val > max_val) val = max_val - 1;
    if (val < min_val) val = min_val + 1;
    uin32 ans = (val - min_val) * (1 << bitwidth) / (max_val - min_val); 
    return ans;
}


// packing weight for the hidden FC layer. STEP128(A_height)*PAD128(A_width)
__global__ void PackFcWeight128(const uin32* __restrict__ A, uin32* B, 
                                    const int A_height, const int A_width, const int w_bit)
{
    GET_LANEID;
    GET_WARPID;

    const int gdx = STEP128(A_height);
    const int gdy = STEP8(A_width);

    const int lx = (warpid & 0x3); // warp x_index vertically
    const int ly = (warpid >> 2);  // warp y_index hozerionsally.

    const int offset_opt = STEP128(A_height)*PAD128(A_width)*128/32;

    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        const int bx = bid % gdx;
        const int by = bid / gdx;
        
        for (int bIdx = 0; bIdx < w_bit; bIdx++){
            float f0 = ( (bx*128+lx*32+laneid<A_height) && (by*8+ly<A_width) )? \
                        ((A[(bx*128+lx*32+laneid)*A_width+by*8+ly]>>bIdx) & 0x01):-1.0f;
            unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0));
            if (laneid==0) 
                B[bIdx*offset_opt + (by*8+ly)*gdx*4+bx*4 + lx] = r0;
        }
    }
}
// compress the input from 32-bit to 1-bit
// store in 1-bit with packed 32-bit unsigned int format.
__global__  void QGTC_layer_input(
    uint32_t* bit_T_out, 
    uint32_t* __restrict__ T_in, 
    const int height,
    const int width,
    const int bitWidth
)
{
    GET_LANEID;
    GET_WARPID;

    // how many blocks in total. X_height/8 * X_width/128
    const int gdx = STEP8(height);                     // x size: vertical.
    const int gdy = STEP128(width);                    // y size: horizontal.
    const int offset = (height)*(width);      // layerwise offset of INPUT before bit compression.
    const int offset_opt = PAD8(height)*STEP128(width)*128/32;        // layerwise offset of OUTPUT after bit compression.

    // 32 warps per block
    const int lx = (warpid >> 2);  // x index, vertical
    const int ly = (warpid & 0x3); // y index, horizontal

    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {

        const int bx = bid / gdy; // x index of the current block
        const int by = bid % gdy; // y index of the current block

        // iterate through all bits
        for (int bitIdx = 0; bitIdx < bitWidth; bitIdx++){
            // boundry check whether inside, otherwise set to -1
            // uint32_t f0 = ( (by*128+ly*32+laneid<(width)) && (bx*8+lx<(height)) )?
            //             T_in[bitIdx*offset + (bx*8+lx)*(width)+by*128+ly*32+laneid]: 0;

            uint32_t f0 = ( (by*128+ly*32+laneid<(width)) && (bx*8+lx<(height)) )?
                            ((T_in[(bx*8+lx)*(width)+by*128+ly*32+laneid]>>bitIdx) & 0x01): 0;

            // compressed, any thing outside boundry would be set to 0.
            // note that * f0 > 0 * in the 0/1 case. but >= 0 in 1/-1 case
            unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0));

            // output the results
            if (laneid==0)
                bit_T_out[bitIdx*offset_opt + (bx*8+lx)*gdy*4 + by*4 + ly] = r0;
        }

    }
}

// (bit_X, bit_W) --> (uint32 bit_X_out)
__global__ void QGTC_layer_hidden(
    uint32_t* bit_X_out, 
    uint32_t* __restrict__ bit_X, 
    uint32_t* __restrict__ bit_W,
    const int X_height,
    const int X_width,
    const int W_width,
    const int w_bit,
    const int act_bit
)
{
    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;

    GET_LANEID;
    GET_WARPID;
    
    // layerwise offset measured in uint32_t
    const int act_offset = PAD8(X_height)*STEP128(X_width)*128/32;
    const int w_offset = STEP128(X_width)*PAD128(W_width)*128/32;
    const int opt_offset = PAD8(X_height)*STEP128(W_width)*128/32;

    // M x N x K
    extern __shared__ int Cs[];
    const int gdx = STEP8(X_height);     // vertical     --> M
    const int gdy = STEP8(W_width);     // horizontal   --> N
    const int gdk = STEP128(X_width);    // iterations   --> K
    const int gdm = STEP128(W_width);   // output width ---> N

    // each grid with gridim.x blocks, 
    // each block with 32 warps.
    // each warp processes each 8x8 tile
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> tmp_frag;

        wmma::fill_fragment(c_frag, 0);

        // rwo major output.
        const int bx = bid / gdy;
        const int by = bid % gdy;

        // iterate along different bits.
        for (int bit = 0; bit < act_bit*w_bit; bit++){
            int b_act = bit % act_bit;
            int b_w = bit / act_bit;
            int b_opt = b_act + b_w;

            // accmuluation of the current bit.
            wmma::fill_fragment(tmp_frag, 0);

            // iterate along the K columns
            for (int i=0; i<gdk; i++)
            {
                // iterate along the K diemsnion by loading the tile from the 
                load_matrix_sync(a_frag, bit_X + b_act*act_offset + bx*8*gdk*4 + i*128/32, gdk*128);
                load_matrix_sync(b_frag, bit_W + b_w*w_offset + by*8*gdk*4 + i*128/32, gdk*128);
                bmma_sync(tmp_frag, a_frag, b_frag, tmp_frag, bmmaBitOpAND);
            }

            // Accumulation.
            #pragma unroll
            for (int t = 0; t < tmp_frag.num_elements; t++) {
                c_frag.x[t] += tmp_frag.x[t] << b_opt;
            }
        }

        // quantization at the fragment into act_bit (stored in uint32).
        #pragma unroll
        for (int t = 0; t < c_frag.num_elements; t++) {
            // printf("%d\n", c_frag.x[t]);
            c_frag.x[t] = quantize(c_frag.x[t], act_bit);
        }

        // finished one output tile and store to shared memory
        store_matrix_sync(&Cs[warpid*64], c_frag, 8, wmma::mem_row_major);


        for (int bIdx = 0; bIdx < act_bit; bIdx++){
            
            // change to 8-bit address
            uin8* Cb = (uin8*)(&(bit_X_out[bIdx*opt_offset])); 

            // 2D index of a warp
            const int gy = (laneid%8);
            const int gx = (laneid/8);

            // checking position constraints.
            bool v0_in = ((by*8+gy)<(W_width)) && ((bx*8+gx)<(X_height));
            bool v1_in = ((by*8+gy)<(W_width)) && ((bx*8+gx+4)<(X_height)); 

            // get the corresponding decomposed bit value.
            bool v0 = v0_in && (((Cs[warpid*64+laneid]>>bIdx) & 0x1) > 0);
            bool v1 = v1_in && (((Cs[warpid*64+32+laneid]>>bIdx) & 0x1) > 0);

            union{ uint32_t data; uin8 elements[4];} p0, p1;

            // pack into 32 1-bit.
            p0.data = __brev(__ballot_sync(0xFFFFFFFF, v0 > 0));
            p1.data = __brev(__ballot_sync(0xFFFFFFFF, v1 > 0));

            // printf("p0.data: %u, p1.data: %u\n", p0.data, p1.data); // ok, all 1s.
            __syncthreads();

            // output to binary after compression.
            if (laneid < 4)
            {
                Cb[(bx*8+laneid)*gdm*16+FLIPBITS(by,2)] = p0.elements[3-laneid]; 
                Cb[(bx*8+4+laneid)*gdm*16+FLIPBITS(by,2)] = p1.elements[3-laneid]; 
            }
        }
        //end
    }
}

//
// (bit_X, bit_W) --> (float X_out)
//
__global__ void QGTC_layer_output(
    float* X_out, 
    uint32_t* __restrict__ bit_X, 
    uint32_t* __restrict__ bit_W,
    const int X_height,
    const int X_width,
    const int W_width,
    const int w_bit,
    const int act_bit
)
{
    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;

    GET_LANEID;
    GET_WARPID;
    extern __shared__ int Cs[];

    const int act_offset = PAD8(X_height)*STEP128(X_width)*128/32;
    const int w_offset = STEP128(X_width)*PAD8(W_width)*128/32;

    const int gdx = STEP8(X_height); //vertical
    const int gdy = STEP8(W_width); //horizontal
    const int gdk = STEP128(X_width);

    // printf("act_bit: %d, w_bit: %d, act_offset: %d, w_offset: %d, gdx: %d, gdy: %d, gdk: %d\n", act_bit, w_bit, act_offset, w_offset, gdx, gdy, gdk);

    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> tmp_frag;

        const int bx = bid / gdy;
        const int by = bid % gdy;

        wmma::fill_fragment(c_frag, 0);

        for (int bit = 0; bit < act_bit*w_bit; bit++){

            int b_act = bit % act_bit;
            int b_w = bit / act_bit;
            int b_opt = b_act + b_w;

            // accmuluation of the current bit.
            wmma::fill_fragment(tmp_frag, 0);

            for (int i=0; i<gdk; i++)
            {
                load_matrix_sync(a_frag, bit_X + b_act*act_offset + bx*8*gdk*4 + i*128/32, gdk*128);
                load_matrix_sync(b_frag, bit_W + b_w*w_offset + by*8*gdk*4 + i*128/32, gdk*128);
                bmma_sync(tmp_frag, a_frag, b_frag, tmp_frag, bmmaBitOpAND);
            }

            // Accumulation.
            #pragma unroll
            for (int t = 0; t < tmp_frag.num_elements; t++) 
            {
                // printf("%d\n", c_frag.x[t]);
                c_frag.x[t] += (tmp_frag.x[t] << b_opt);
            }
            __syncwarp();
        }

        store_matrix_sync(&Cs[warpid*64], c_frag, 8, wmma::mem_row_major);
        // if (laneid == 0 && warpid == 0){
        //     for (int i=0; i<8; i++){
        //         for (int j=0; j<8; j++){
        //             printf("%u ", Cs[warpid*64 + i * 8 + j]);
        //         }
        //         printf("\n");
        //     }
        // }

        float* output_sub = &(X_out[bx*(W_width)*8+by*8]);

        if (laneid < 8)
        {
            for (int j=0; j<8; j++)
            {
                if ((bx*8+j)<(X_height))
                {
                    if (by*8+laneid<(W_width))
                    {
                        float val = Cs[warpid*64+j*8+laneid] * 1.0f; //* (p->bn_scale_gpu[by*8+laneid]) + (p->bn_bias_gpu[by*8+laneid]);
                        output_sub[j*(W_width)+laneid] = val;
                    }
                }
            }
        } //end
    }
}
#endif