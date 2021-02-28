import os
import QGTC
import math
import sys
import torch
import torch.nn as nn


class Aggregation_Qnt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bit_A, bit_X, bit_W, act_bit, w_bit, input=False, hidden=False, output=False):
        # ctx.save_for_backward(X, )

        assert input+hidden+output == 1

        if input or hidden:
            out = QGTC.mm_v1(bit_X, bit_W, act_bit, w_bit)
            out = QGTC.mm_v1(bit_X, bit_W, 1, act_bit)
        
        if output:
            out = QGTC.mm_v1(bit_X, bit_W, act_bit, w_bit)
            out = QGTC.mm_v1(bit_X, bit_W, 1, act_bit)

        return out

    @staticmethod
    def backward(ctx, d_output):
        pass
        return None, None, None, None
        # X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = ctx.saved_tensors

        # # SPMM backward propagation.
        # d_input_prime = GAcc.forward(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        # # GEMM backward propagation.
        # d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        # d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        # return d_input, d_weights, None, None, None, None, None, None

class GCNConv_Qnt(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, w_bit=2, act_bit=3):
        super(GCNConv, self).__init__()

        self.W_in = torch.nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_out = torch.nn.Parameter(torch.randn(hidden_dim, output_dim))

        self.w_bit = w_bit
        self.act_bit = act_bit

        # weight dev_ptr for low-bit weight [w_bit, H, W]
        self.bit_W_in = None
        self.bit_W_out = None

        # Adjacent dev_ptr for 1-bit adjacent matrix [N, N]
        self.bit_A = None
        self.bit_A = None

    def weight_Qnt(self):
        self.bit_W_in = QGTC.bit_qnt(self.W_in, self.w_bit)
        self.bit_W_out = QGTC.bit_qnt(self.W_out, self.w_bit)

    def A_Qnt(self, A):
        return QGTC.bit_qnt(A, 1)
    
    def X_Qnt(self, X):
        return QGTC.bit_qnt(X, self.act_bit)

    def forward(self, A, X):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the dense format of matrix, shape: [n_nodes, n_nodes] for a subgraph.
        '''
        bit_A = self.A_Qnt(A)
        bit_X_in = self.X_Qnt(X)
        
        bit_out = Aggregation_Qnt.apply(bit_A, bit_X_in, self.bit_W_in, input=True)
        out = Aggregation_Qnt.apply(bit_A, bit_out, self.bit_W_out, output=True)

        return out



class Aggregation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, X, W):
        # ctx.save_for_backward(X, )
        # GEMM node update
        out = torch.mm(X, W)
        out = torch.mm(A, out)
        # X_prime = QGTC.forward(out)[0]
        return out

    @staticmethod
    def backward(ctx, d_output):
        pass
        return None, None, None
        # X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = ctx.saved_tensors

        # # SPMM backward propagation.
        # d_input_prime = GAcc.forward(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        # # GEMM backward propagation.
        # d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        # d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        # return d_input, d_weights, None, None, None, None, None, None

class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GCNConv, self).__init__()

        self.W_in = torch.nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_out = torch.nn.Parameter(torch.randn(hidden_dim, output_dim))

    def forward(self, A, X):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the dense format of matrix, shape: [n_nodes, n_nodes] for a subgraph.
        '''
        X_h1 = Aggregation.apply(A, X, self.W_in)
        out = Aggregation.apply(A, X_h1, self.W_out)

        return out