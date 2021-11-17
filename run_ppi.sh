#!/bin/bash

# python cluster_gcn.py --gpu 0 --dataset ppi --lr 1e-2 --weight-decay 0.0 --psize 10 --batch-size 1 --n-epochs 10 \
#   --n-hidden 2048 --n-layers 3 --log-every 100 --use-pp --self-loop \
#   --note self-loop-ppi-non-sym-ly3-pp-cluster-2-2-wd-0 --dropout 0.2 --use-val --normalize

python cluster_gcn.py --gpu 0 --dataset ppi --use_QGTC --run_GIN