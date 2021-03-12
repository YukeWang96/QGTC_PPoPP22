import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import time
import random
import sys

import numpy as np
import networkx as nx
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
import collections
import os.path as osp

from modules import *
from sampler import ClusterIter
from utils import Logger, evaluate, save_log_dir, load_data

import matplotlib.pylab as plt
import numpy as np
from scipy.sparse import coo_matrix

# from QGTC_conv import *
import QGTC
from dataset import *
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AMDataset, AmazonCoBuyComputerDataset

from config import *

def PAD8(input):
    return int((input + 7)//8)

def PAD128(input):
    return int((input + 127)//128)

def main(args):
    torch.manual_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    multitask_data = set(['ppi'])
    multitask = args.dataset in multitask_data

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
        path = osp.join("/home/yuke/.graphs/orig", args.dataset)
        data = QGTC_dataset(path, args.dim, args.n_classes)
        g = data.g
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    psize = len(train_mask)/args.psize
    # print(train_mask)
    # sys.exit(0)
    train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)

    # Normalize features
    # if args.normalize:
    #     feats = g.ndata['feat']
    #     train_feats = feats[train_mask]
    #     scaler = sklearn.preprocessing.StandardScaler()
    #     scaler.fit(train_feats.data.numpy())
    #     features = scaler.transform(feats.data.numpy())
    #     g.ndata['feat'] = torch.FloatTensor(features)

    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_classes
    n_edges = g.number_of_edges()
    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    # print("""----Data statistics------'
    # #Edges %d
    # #Classes %d
    # #Train samples %d
    # #Val samples %d
    # #Test samples %d""" %
    #         (n_edges, n_classes,
    #         n_train_samples,
    #         n_val_samples,
    #         n_test_samples))

    # create GCN model
    # if args.self_loop and not args.dataset.startswith('reddit'):
        # g = dgl.remove_self_loop(g)
        # g = dgl.add_self_loop(g)
        # print("adding self-loop edges")

    # metis only support int64 graph
    g = g.long()
    # get the subgraph based on the partitioning nodes list.
    cluster_iterator = ClusterIter(args.dataset, g, args.psize, args.batch_size, train_nid, use_pp=False)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        g = g.int().to(args.gpu)

    # print('labels shape:', g.ndata['label'].shape)
    # print("features shape, ", g.ndata['feat'].shape)
    feat_size  = g.ndata['feat'].shape[1]

    if use_PyG:
        model = SAGE_PyG(in_feats, args.n_hidden, 
                            n_classes, num_layers=args.n_layers+2)
    else:
        model = GraphSAGE(in_feats, args.n_hidden,
                        n_classes, args.n_layers, F.relu,
                        args.dropout, args.use_pp)
    # print(model)
    if cuda:
        model.cuda()

    # logger and so on
    # log_dir = save_log_dir(args)
    # logger = Logger(os.path.join(log_dir, 'loggings'))
    # logger.write(args)

    # Loss function
    if multitask:
        # print('Using multi-label loss')
        loss_f = nn.BCEWithLogitsLoss()
    else:
        # print('Using multi-class loss')
        loss_f = nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)


    # print("\n\n==> subgraph-size: {:.2f}".format(psize))

    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()
        # print("current memory after model before training",
        #       torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
        # print("---------------------------------\n\n")

    start_time = time.time()
    best_f1 = -1

    hidden_1 = 128
    # hidden_2 = 2048
    output = args.n_classes

    # total_ops = 0
    allocation = 0
    running_time = 0

    W_1 = torch.ones((feat_size, hidden_1)).cuda()
    W_2 = torch.ones((hidden_1, hidden_1)).cuda()
    W_3 = torch.ones((hidden_1, output)).cuda()

    bw_A = 1
    bw_X = 3
    bw_W = 3

    bit_W1 = QGTC.bit_qnt(W_1.cuda(), bw_W, True, False)
    bit_W2 = QGTC.bit_qnt(W_2.cuda(), bw_W, True, False)
    bit_W3 = QGTC.bit_qnt(W_3.cuda(), bw_W, True, True)
    # model = GCNConv(feat_size*2, hidden_1, output).cuda()

    cnt = 0
    for epoch in range(args.n_epochs):
        for j, cluster in enumerate(cluster_iterator):
            # sync with upper level training graph      
            if regular:
                torch.cuda.synchronize()
                t = time.perf_counter()      
                if cuda:
                    cluster = cluster.to(torch.cuda.current_device())

                torch.cuda.synchronize()
                allocation += time.perf_counter() - t
                # model.train()
                torch.cuda.synchronize()
                t = time.perf_counter()   
                
                if use_PyG:
                    # print(cluster.edges())
                    edge_idx = torch.stack([cluster.edges()[0],cluster.edges()[1]], dim=0).long()
                    # print(edge_idx.size())
                    # print(cluster.ndata['feat'].size())
                    model(cluster.ndata['feat'], edge_idx)
                else:
                    pred = model(cluster)       # model training

                torch.cuda.synchronize()
                running_time += time.perf_counter() - t
                num_nodes = len(cluster.nodes())

            else:
                # torch.cuda.synchronize()
                t = time.perf_counter()

                # version-1
                cluster = cluster.cuda()
                A = cluster.A.to_dense()            
                X = cluster.X

                # # version-2
                # # unzip adjacent matrix A
                # A = cluster.A.to_dense().cuda()            
                # # unzip feature embedding matrix X
                # X = cluster.X.cuda()

                # torch.cuda.synchronize()
                allocation += time.perf_counter() - t

                # torch.cuda.synchronize()
                t = time.perf_counter()
                
                if use_QGTC:
                    # 1-layer
                    bit_A = QGTC.bit_qnt(A, bw_A, False, False)
                    bit_X = QGTC.bit_qnt(X, bw_X, True, False)

                    if epoch == 0 and j == 0:
                        print("A.size: {}".format(A.size()))

                    # 1-layer
                    bit_output = QGTC.mm_v1(bit_A, bit_X, A.size(0), A.size(0), X.size(1), bw_A, bw_X, bw_X)
                    bit_output_1 = QGTC.mm_v1(bit_output, bit_W1, A.size(0), X.size(1), W_1.size(1), bw_X, bw_W, bw_X)

                    # 2-layer
                    bit_output_2 = QGTC.mm_v1(bit_A, bit_output_1, A.size(0), A.size(0), W_1.size(1), bw_A, bw_X, bw_X)
                    bit_output_3 = QGTC.mm_v1(bit_output_2, bit_W2, A.size(0), W_1.size(1), W_2.size(1), bw_X, bw_W, bw_X)

                    # 3-layer
                    bit_output_4 = QGTC.mm_v1(bit_A, bit_output_3, A.size(0), A.size(0), W_2.size(1), bw_A, bw_X, bw_X)
                    float_output = QGTC.mm_v2(bit_output_4, bit_W3, A.size(0), W_2.size(1), W_3.size(1), bw_X, bw_W)

                    del bit_A
                    del bit_X
                    del bit_output
                    del bit_output_1
                    del bit_output_2
                    del bit_output_3
                    del bit_output_4
                    del float_output
                    torch.cuda.empty_cache()
                else:
                    # 1-layer
                    X = torch.mm(A, X)
                    X_out = torch.mm(X, W_1)
                    # 2-layer
                    X_out = torch.mm(A, X_out)
                    X_out = torch.mm(X_out, W_2)
                    # 3-layer
                    X_out = torch.mm(A, X_out)
                    X_out = torch.mm(X_out, W_3)

                # torch.cuda.synchronize()
                running_time += time.perf_counter() - t

                # num_nodes = A.size(0)
                # total_ops += 2*num_nodes*num_nodes*hidden_1 +  2*num_nodes*feat_size*hidden_1 \
                #             + 2*num_nodes*num_nodes*output + 2*num_nodes*hidden_1*output

            # batch_labels = cluster.ndata['label']
            # batch_train_mask = cluster.ndata['train_mask']
            # loss = loss_f(pred[batch_train_mask],
            #               batch_labels[batch_train_mask])

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # if j % args.log_every == 0:
            #     print("epoch:{}/{}, Iteration {}/{}, #N: {}, {:.3f} GB"\
            #     .format(epoch, args.n_epochs, j, len(cluster_iterator), num_nodes, \
            #         torch.cuda.memory_allocated(device=cluster.device) / 1024 / 1024 / 1024)),
            # print("{}".format(epoch), end=" ")
        cnt += 1
        print("Epoch: {}".format(epoch))
        # hand the current tensor back to host Memory
        # cluster = cluster.cpu()

    # torch.cuda.synchronize()
    end_time = time.time()
    print("allocation: {:.3f} ms, inference: {:.3f} ms".format(allocation/cnt*1e3, running_time/cnt*1e3))
    print("Avg. Epoch: {:.3f} ms".format((end_time - start_time)*1000/cnt))
    # print("GFLOPS: {:.3f}".format(total_ops/(end_time - start_time) / 10e9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
                        help="learning rate")
    parser.add_argument("--dim", type=int, default=10,
                        help="input dimension of each dataset")

    parser.add_argument("--n-epochs", type=int, default=3, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=20, help="batch size")
    parser.add_argument("--psize", type=int, default=1500, help="partition number")

    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--log-every", type=int, default=100,
                        help="the frequency to save model")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="test batch size")
    parser.add_argument("--n-classes", type=int, default=10,
                        help="number of classes")

    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--val-every", type=int, default=1,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--rnd-seed", type=int, default=3,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--use-pp", action='store_true',
                        help="whether to use precomputation")
    parser.add_argument("--normalize", action='store_true',
                        help="whether to use normalized feature")
    parser.add_argument("--use-val", action='store_true',
                        help="whether to use validated best model to test")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--note", type=str, default='none',
                        help="note for log dir")

    args = parser.parse_args()
    # print(args)
    main(args)
