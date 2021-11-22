import argparse
import time
import random
import os.path as osp
import numpy as np

import torch
from dgl.data import register_data_args
from modules import *
from sampler import ClusterIter
from utils import load_data
from tqdm import *
import numpy as np
import QGTC
from dataset import *
from ogb.nodeproppred import DglNodePropPredDataset
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
    hidden_1 = args.n_hidden
    output = args.n_classes

    # total_ops = 0
    transfering = 0
    running_time = 0

    W_1 = torch.ones((feat_size, hidden_1)).cuda()
    W_2 = torch.ones((hidden_1, hidden_1)).cuda()
    W_3 = torch.ones((hidden_1, output)).cuda()

    bw_A = 1
    bw_X = 8
    bw_W = bw_X

    bit_W1 = QGTC.val2bit(W_1.cuda(), bw_W, True, False)
    bit_W2 = QGTC.val2bit(W_2.cuda(), bw_W, True, False)
    bit_W3 = QGTC.val2bit(W_3.cuda(), bw_W, True, True)

    layer1_t = 0
    layer2_t = 0
    layer3_t = 0

    cnt = 0
    for epoch in tqdm(range(args.n_epochs)):
        for j, cluster in enumerate(cluster_iterator):
            
            # for DGL and PyG
            if args.regular: 
                torch.cuda.synchronize()
                t = time.perf_counter()      
                cluster = cluster.to(torch.cuda.current_device())
                torch.cuda.synchronize()
                transfering += time.perf_counter() - t
                
                torch.cuda.synchronize()
                t = time.perf_counter()   
                if args.use_PyG:
                    edge_idx = torch.stack([cluster.edges()[0], cluster.edges()[1]], dim=0).long()
                    model(cluster.ndata['feat'], edge_idx)
                else:
                    model(cluster)    
                torch.cuda.synchronize()
                running_time += time.perf_counter() - t
            else:
                torch.cuda.synchronize()
                t = time.perf_counter()

                ## version-1 seperate, dense edge, dense node embedding
                # A = cluster.A.to_dense().cuda()            
                # X = cluster.X.cuda()

                # # # version-2 seperate, sparse edge, dense node embedding.
                # A = cluster.A.cuda().to_dense()            
                # X = cluster.X.cuda()

                # #@ version-3 packed, sparse edge, dense node embedding.
                cluster = cluster.cuda()
                A = cluster.A.to_dense()            
                X = cluster.X

                torch.cuda.synchronize()
                transfering += time.perf_counter() - t

                torch.cuda.synchronize()
                t = time.perf_counter()
                
                if args.use_QGTC:
                    if args.run_GIN:
                        # torch.cuda.synchronize()
                        # t = time.perf_counter()
                        # 1-layer [in_feat, hidden]
                        # print("A.size: {}".format(A.size()))
                        # print("X.size: {}".format(X.size()))

                        bit_A = QGTC.val2bit(A, bw_A, False, False)
                        bit_X = QGTC.val2bit(X, bw_X, True, False)
                        bit_output = QGTC.bitMM2Bit(bit_A, bit_X, A.size(0), A.size(0), X.size(1), bw_A, bw_X, bw_X)
                        bit_output_1 = QGTC.bitMM2Bit(bit_output, bit_W1, A.size(0), X.size(1), W_1.size(1), bw_X, bw_W, bw_X)
                        # torch.cuda.synchronize()
                        # layer1_t += time.perf_counter() - t

                        # 2-layer  [hidden, hidden]
                        # torch.cuda.synchronize()
                        # t = time.perf_counter()
                        bit_output_2 = QGTC.bitMM2Bit(bit_A, bit_output_1, A.size(0), A.size(0), W_1.size(1), bw_A, bw_X, bw_X)
                        bit_output_3 = QGTC.bitMM2Bit(bit_output_2, bit_W2, A.size(0), W_1.size(1), W_2.size(1), bw_X, bw_W, bw_X)
                        # torch.cuda.synchronize()
                        # layer2_t += time.perf_counter() - t

                        # 3-layer  [hidden, output]
                        # torch.cuda.synchronize()
                        # t = time.perf_counter()
                        bit_output_4 = QGTC.bitMM2Bit(bit_A, bit_output_3, A.size(0), A.size(0), W_2.size(1), bw_A, bw_X, bw_X)
                        float_output = QGTC.bitMM2Int(bit_output_4, bit_W3, A.size(0), W_2.size(1), W_3.size(1), bw_X, bw_W, False)
                        # torch.cuda.synchronize()
                        # layer3_t += time.perf_counter() - t
                    
                    else: # GCN
                        bit_A = QGTC.val2bit(A, bw_A, False, False)
                        bit_X = QGTC.val2bit(X, bw_X, True, False)
                        
                        if args.zerotile_jump:
                            QGTC.bitMM2Bit_base_cnt(bit_X, bit_W1, X.size(0), X.size(1), W_1.size(1), bw_X, bw_W, bw_X)
                            QGTC.bitMM2Bit_zerojump_cnt(bit_X, bit_W1, X.size(0), X.size(1), W_1.size(1), bw_X, bw_W, bw_X)
                            continue
                            
                        # 1-layer [in_feat, hidden]
                        bit_output = QGTC.bitMM2Bit(bit_X, bit_W1, X.size(0), X.size(1), W_1.size(1), bw_X, bw_W, bw_X)
                        bit_output_1 = QGTC.bitMM2Bit(bit_A, bit_output, A.size(0), A.size(1), W_1.size(1), bw_A, bw_X, bw_X)

                        # 2-layer  [hidden, hidden]
                        bit_output_2 = QGTC.bitMM2Bit(bit_output_1, bit_W2, A.size(0), W_1.size(1), W_2.size(1), bw_X, bw_W, bw_X)
                        bit_output_3 = QGTC.bitMM2Bit(bit_A, bit_output_2, A.size(0), A.size(0), W_2.size(1), bw_A, bw_X, bw_X)

                        # 3-layer  [hidden, output]
                        bit_output_4 = QGTC.bitMM2Bit(bit_output_3, bit_W3, A.size(0), W_2.size(1), W_3.size(1), bw_X, bw_W, bw_X)
                        float_output = QGTC.bitMM2Int(bit_A, bit_output_4, A.size(0), A.size(0), W_2.size(1), bw_A, bw_X, False)

                    # del bit_A
                    # del bit_X
                    # del bit_output
                    # del bit_output_1
                    # del bit_output_2
                    # del bit_output_3
                    # del bit_output_4
                    # del float_output
                    # torch.cuda.empty_cache()
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
