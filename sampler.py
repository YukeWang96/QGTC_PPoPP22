import os
import random

import dgl.function as fn
import torch

from partition_utils import *
import sys
from scipy.sparse import coo_matrix

regular=False

class ClusterTensor(torch.nn.Module):
    def __init__(self, A, X):
        super(ClusterTensor, self).__init__()
        self.A =  torch.nn.Parameter(A)
        self.X =  torch.nn.Parameter(X)
    
    def forward(self):
        pass

class ClusterIter(object):
    '''The partition sampler given a DGLGraph and partition number.
    The metis is used as the graph partition backend.
    '''
    def __init__(self, dn, g, psize, batch_size, seed_nid, use_pp=True):
        """Initialize the sampler.

        Paramters
        ---------
        dn : str
            The dataset name.
        g  : DGLGraph
            The full graph of dataset
        psize: int
            The partition number
        batch_size: int
            The number of partitions in one batch
        seed_nid: np.ndarray
            The training nodes ids, used to extract the training graph
        use_pp: bool
            Whether to use precompute of AX
        """
        self.use_pp = use_pp
        self.g = g.subgraph(seed_nid)

        # print(use_pp)
        # precalc the aggregated features from training graph only
        if use_pp:
            self.precalc(self.g)
            print('precalculating')

        self.psize = psize
        self.batch_size = batch_size
        # cache the partitions of known datasets&partition number
        if dn:
            fn = os.path.join('./datasets/', dn + '_{}.npy'.format(psize))
            if os.path.exists(fn):
                self.par_li = np.load(fn, allow_pickle=True)
            else:
                os.makedirs('./datasets/', exist_ok=True)
                self.par_li = get_partition_list(self.g, psize)
                np.save(fn, self.par_li)
        else:
            self.par_li = get_partition_list(self.g, psize)
        
        self.max = int((psize) // batch_size)
        random.shuffle(self.par_li)
        self.get_fn = get_subgraph

        if not regular:
            self.cTensor_li = []
            # preprocess all subgraphs.
            for cid in range(self.max):
                cluster = self.get_fn(self.g, self.par_li, cid, self.psize, self.batch_size)
                
                num_nodes = len(cluster.nodes())
                edges = cluster.edges()
                row  = edges[0].numpy()
                col  = edges[1].numpy()
                data = np.ones(len(row))
                X = cluster.ndata['feat']
                indices = np.vstack((row, col))

                i = torch.LongTensor(indices)
                v = torch.FloatTensor(data)
                A = torch.sparse.FloatTensor(i, v, torch.Size((num_nodes, num_nodes)))
                X = torch.FloatTensor(X)
                cTensor = ClusterTensor(A, X)
                self.cTensor_li.append(cTensor)

    def precalc(self, g):
        norm = self.get_norm(g)
        g.ndata['norm'] = norm
        features = g.ndata['feat']
        print("features shape, ", features.shape)
        with torch.no_grad():
            g.update_all(fn.copy_src(src='feat', out='m'),
                         fn.sum(msg='m', out='feat'),
                         None)
            pre_feats = g.ndata['feat'] * norm
            # use graphsage embedding aggregation style
            g.ndata['feat'] = torch.cat([features, pre_feats], dim=1)

    # use one side normalization
    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.g.ndata['feat'].device)
        return norm

    def __len__(self):
        return self.max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        global regular
        if self.n < self.max:
            if not regular:
                item = self.cTensor_li[self.n]
                self.n += 1
                return item
            else:
                result = self.get_fn(self.g, self.par_li, self.n,
                                    self.psize, self.batch_size)
                self.n += 1
                return result
        else:
            random.shuffle(self.par_li)
            raise StopIteration
