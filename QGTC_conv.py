import os
# import QGTC
import math
import sys
import torch
import torch.nn as nn


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