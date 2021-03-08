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

from modules import GraphSAGE
from sampler import ClusterIter
from utils import Logger, evaluate, save_log_dir, load_data

import matplotlib.pylab as plt
import numpy as np
from scipy.sparse import coo_matrix

from QGTC_conv import *
import QGTC


regular = False

def main(args):
    torch.manual_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    multitask_data = set(['ppi'])
    multitask = args.dataset in multitask_data

    # load and preprocess dataset
    data = load_data(args)
    g = data.g
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']

    psize = len(train_mask)/args.psize
    train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)

    # Normalize features
    if args.normalize:
        feats = g.ndata['feat']
        train_feats = feats[train_mask]
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_feats.data.numpy())
        features = scaler.transform(feats.data.numpy())
        g.ndata['feat'] = torch.FloatTensor(features)

    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_classes
    n_edges = g.number_of_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
            (n_edges, n_classes,
            n_train_samples,
            n_val_samples,
            n_test_samples))

    # create GCN model
    if args.self_loop and not args.dataset.startswith('reddit'):
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        print("adding self-loop edges")

    # metis only support int64 graph
    g = g.long()

    # get the subgraph based on the partitioning nodes list.
    cluster_iterator = ClusterIter(args.dataset, g, args.psize, args.batch_size, train_nid, use_pp=args.use_pp)

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        g = g.int().to(args.gpu)

    print('labels shape:', g.ndata['label'].shape)
    print("features shape, ", g.ndata['feat'].shape)
    feat_size  = g.ndata['feat'].shape[1]

    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.use_pp)


    if cuda:
        model.cuda()

    # logger and so on
    log_dir = save_log_dir(args)
    logger = Logger(os.path.join(log_dir, 'loggings'))
    logger.write(args)

    # Loss function
    if multitask:
        print('Using multi-label loss')
        loss_f = nn.BCEWithLogitsLoss()
    else:
        print('Using multi-class loss')
        loss_f = nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)


    print("\n\n==> subgraph-size: {}\n".format(psize))
    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()
        print("current memory after model before training",
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
        print("---------------------------------\n\n")

    start_time = time.time()
    best_f1 = -1

    hidden_1 = 128
    # hidden_2 = 2048
    output = 42

    total_ops = 0
    allocation = 0
    running_time = 0

    W_1 = torch.ones((feat_size*2, hidden_1)).cuda()
    # W_2 = torch.ones((hidden_1, output)).cuda()

    bw_A = 1
    bw_W1 = 1
    bw_X = 1
    # model = GCNConv(feat_size*2, hidden_1, output).cuda()
    bit_W1 = QGTC.bit_qnt(W_1.cuda(), bw_W1, True)

    for epoch in range(args.n_epochs):
        cnt = 0
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

                pred = model(cluster)
                
                torch.cuda.synchronize()
                running_time += time.perf_counter() - t
                num_nodes = len(cluster.nodes())

            else:
                torch.cuda.synchronize()
                t = time.perf_counter()
                cluster = cluster.cuda()
                A = cluster.A.to_dense()
                X = cluster.X
                torch.cuda.synchronize()
                allocation += time.perf_counter() - t
                
                torch.cuda.synchronize()
                t = time.perf_counter()
                
                bit_A = QGTC.bit_qnt(A.cuda(), bw_A, False)
                bit_X = QGTC.bit_qnt(X.cuda(), bw_X, True)
                bit_output = QGTC.mm_v1(bit_A, bit_X, len(A), len(A), len(X[0]), bw_A, bw_X, bw_X)
                float_output = QGTC.mm_v2(bit_output, bit_W1, len(A), len(X[0]), len(W_1[0]), bw_X, bw_W1)

                # X = torch.mm(A, X)
                # X_out = torch.mm(X, W_1)

                # X_out = torch.mm(A, X_out)
                # X_out = torch.mm(X_out, W_2)
                # X_out = torch.mm(A, X_out)
                torch.cuda.synchronize()
                running_time += time.perf_counter() - t

                num_nodes = A.size(0)
                total_ops += 2*num_nodes*num_nodes*hidden_1 +  2*num_nodes*feat_size*hidden_1 \
                            + 2*num_nodes*num_nodes*output + 2*num_nodes*hidden_1*output

            # batch_labels = cluster.ndata['label']
            # batch_train_mask = cluster.ndata['train_mask']
            # loss = loss_f(pred[batch_train_mask],
            #               batch_labels[batch_train_mask])

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            if j % args.log_every == 0:
                print("epoch:{}/{}, Iteration {}/{}, #N: {}, {:.3f} GB"\
                .format(epoch, args.n_epochs, j, len(cluster_iterator), num_nodes, \
                    torch.cuda.memory_allocated(device=A.device) / 1024 / 1024 / 1024)),
            cnt += 1
            
        # hand the current tensor back to host Memory
        cluster = cluster.cpu()


    end_time = time.time()

    print("allocation: {:.3f} ms, inference: {:.3f} ms".format(allocation/cnt*1e3, running_time/cnt*1e3))
    print("Avg. Epoch: {:.3f} ms".format((end_time - start_time)*1000/cnt))
    print("GFLOPS: {:.3f}".format(total_ops/(end_time - start_time) / 10e9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--log-every", type=int, default=100,
                        help="the frequency to save model")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="batch size")
    parser.add_argument("--psize", type=int, default=1500,
                        help="partition number")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="test batch size")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
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

    print(args)

    main(args)
