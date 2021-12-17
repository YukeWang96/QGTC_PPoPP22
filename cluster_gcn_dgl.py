import argparse
import time
import random
import os.path as osp
import numpy as np

import torch
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import register_data_args

from modules import *
from sampler import ClusterIter
from utils import load_data
from dataset import *
from tqdm import *
import QGTC

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
register_data_args(parser)
parser.add_argument("--gpu", type=int, default=0, help="gpu")

parser.add_argument("--n-epochs", type=int, default=20, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=20, help="batch size")
parser.add_argument("--psize", type=int, default=1500, help="number of partitions")

parser.add_argument("--dim", type=int, default=10, help="input dimension of each dataset")
parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
parser.add_argument("--n-classes", type=int, default=10, help="number of classes")
parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")

parser.add_argument("--use-pp", action='store_true',help="whether to use precomputation")
parser.add_argument("--regular", action='store_true',help="whether to use PyG or DGL")
parser.add_argument("--use_PyG", action='store_true',help="whether to use PyG")
parser.add_argument("--run_GIN", action='store_true',help="whether to run GIN model")
parser.add_argument("--use_QGTC", action='store_true',help="whether to use QGTC")
parser.add_argument("--zerotile_jump", action='store_true',help="whether to profile zero-tile jumping")

args = parser.parse_args()
print(args)


def main(args):
    torch.manual_seed(3)
    np.random.seed(2)
    random.seed(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load and preprocess dataset
    if args.dataset in ['ppi', 'reddit']:
        data = load_data(args)
        g = data.g
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        labels = g.ndata['label']
    elif args.dataset in ['ogbn-arxiv', 'ogbn-products']:
        data = DglNodePropPredDataset(name=args.dataset) #'ogbn-proteins'
        split_idx = data.get_idx_split()
        g, labels = data[0]
        train_mask = split_idx['train']
        val_mask = split_idx['valid']
        test_mask = split_idx['test']
    else:
        path = osp.join("./qgtc_graphs", args.dataset+".npz")
        data = QGTC_dataset(path, args.dim, args.n_classes)
        g = data.g
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_classes
    # metis only support int64 graph
    g = g.long()
    # get the subgraph based on the partitioning nodes list.
    cluster_iterator = ClusterIter(args.dataset, g, args.psize, args.batch_size, train_nid, use_pp=False, regular=args.regular)

    torch.cuda.set_device(args.gpu)
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()
    g = g.int().to(args.gpu)
    # print('labels shape:', g.ndata['label'].shape)
    # print("features shape, ", g.ndata['feat'].shape)
    feat_size  = g.ndata['feat'].shape[1]

    if args.use_PyG:
        model = SAGE_PyG(in_feats, args.n_hidden, 
                            n_classes, num_layers=args.n_layers+2)
    else:
        if args.run_GIN:
            model = GIN(in_feats, args.n_hidden, n_classes)
        else:
            model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers)

    model.cuda()
    train_nid = torch.from_numpy(train_nid).cuda()

    start_time = time.time()

    transfering = 0
    running_time = 0

    cnt = 0
    for epoch in tqdm(range(args.n_epochs)):
        for j, cluster in enumerate(cluster_iterator):
            # for DGL
            if args.regular: 
                torch.cuda.synchronize()
                t = time.perf_counter()      
                cluster = cluster.to(torch.cuda.current_device())
                torch.cuda.synchronize()
                transfering += time.perf_counter() - t
                
                torch.cuda.synchronize()
                t = time.perf_counter()   
                model(cluster)    # DGL compute
                torch.cuda.synchronize()
                running_time += time.perf_counter() - t

        cnt += 1
        cluster = cluster.cpu()

    torch.cuda.synchronize()
    end_time = time.time()
    print("Trans (ms): {:.3f}, Compute (ms): {:.3f}".format(transfering/cnt*1e3, running_time/cnt*1e3))
    print("Avg. Epoch: {:.3f} ms".format((end_time - start_time)*1000/cnt))

if __name__ == '__main__':
    main(args)
