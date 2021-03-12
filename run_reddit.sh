#!/bin/bash


# python cluster_gcn.py --gpu 0 --dataset reddit-self-loop --lr 1e-2 --weight-decay 0.0 --psize 1500 --batch-size 20 \
#   --n-epochs 30 --n-hidden 128 --n-layers 1 --dropout 0.2 --use-val --normalize

python cluster_gcn.py --gpu 0 --dataset reddit
