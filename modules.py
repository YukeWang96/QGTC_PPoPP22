import math

import dgl.function as fn
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm
import torch.nn.functional as F

########################
### GIN
########################

class GraphSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats):
        super(GraphSAGELayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g = g.local_var()
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), \
                    fn.sum(msg='m', out='h'))
        ah = g.ndata.pop('h')
        h = self.linear(ah)
        h = F.relu(h)
        return h

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 use_pp=False):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphSAGELayer(in_feats, n_hidden))
        # hidden layers
        for i in range(1):
            self.layers.append(
                GraphSAGELayer(n_hidden, n_hidden))
        # output layer
        self.layers.append(GraphSAGELayer(n_hidden, n_classes))

    def forward(self, g):
        h = g.ndata['feat']
        for layer in self.layers:
            h = layer(g, h)
        return h
        
########################
### GIN
########################
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP"""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return h

class GIN(nn.Module):
    """GIN model"""
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                output_dim, 
                num_layers=3):
        """model parameters setting
        Paramters
        ---------
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.ginlayers = torch.nn.ModuleList()
        
        for layer in range(self.num_layers):
            if layer == 0:
                mlp = nn.Linear(input_dim, hidden_dim)
            elif layer < self.num_layers - 1:
                mlp = nn.Linear(hidden_dim, hidden_dim) 
            else:
                mlp = nn.Linear(hidden_dim, output_dim) 

            self.ginlayers.append(GINConv(ApplyNodeFunc(mlp), "sum", init_eps=0, learn_eps=False))

    def forward(self, g):
        h = g.ndata['feat']
        for i in range(self.num_layers):
            h = self.ginlayers[i](g, h)
        return h


class SAGE_PyG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE_PyG, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        return self.convs[-1](x, edge_index)
