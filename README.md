Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
============


Dependencies
------------
- Python 3.7+.
- PyTorch 1.5.0+.
- Deep Graph Library.
- Install QGTC. Go to `QGTC_kernel/`, then run 
```
TORCH_CUDA_ARCH_LIST="8.6" python setup.py  clean --all install 
```

Running Experiments
------------
+ `./bench.py` for running `proteins`, `artist` and `soc-Blogcatalog` dataset.
+ `./run_ppi.sh` for running `PPI` dataset.
+ `./run_ogb.sh` for running `ogbn-arxiv` and `ogbn-products` dataset.
+ `./run_all` for running all the above three experiments together by following the order of `bench.py`, `run_ppi.sh` and `run_ogb.sh`
