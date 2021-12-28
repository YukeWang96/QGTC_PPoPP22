QGTC: Accelerating Quantized GNN via GPU Tensor Core
============

+ **Cite this project and [paper](https://arxiv.org/abs/2111.09547).**
```
@inproceedings{QGTC,
  title={QGTC: Accelerating Quantized GNN via GPU Tensor Core},
  author={Yuke Wang and Boyuan Feng and Yufei Ding},
  booktitle={ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming. (PPoPP'22)},
  year={2022}
}
```


## Clone this project.
------------

```
git clone git@github.com:YukeWang96/PPoPP22_QGTC.git
```

## Dependencies.
------------
- Python 3.7+.
- PyTorch 1.5.0+.
- Deep Graph Library.

## Environment Setup.
------------
### [**Method-1**] Install via Docker (**Recommended**).

+ (i)  Pull docker image:  
```
docker pull happy233/qgtc:updated
docker run -it --rm --gpus all -v $PWD/:/qgtc happy233/qgtc:updated /bin/bash
```
+ (ii) Build docker from scratch:
```
cd Docker/
./build.sh
./launch.sh
```
### [**Method-2**] Install via Conda.
+ Install **`conda`** on system **[Toturial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)**.
+ Create a **`conda`** environment: 
```
conda create -n env_name python=3.6
```
+ Install **`Pytorch`**: 
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```
or using `pip` [**Note that make sure the `pip` you use is the `pip` from current conda environment. You can check this by `which pip`**]
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
+ Install [**`Deep Graph Library (DGL)`**](https://github.com/dmlc/dgl).
```
conda install -c dglteam dgl-cuda11.0
pip install torch requests
```

### Install QGTC. Go to `QGTC_module/`, then run 
```
TORCH_CUDA_ARCH_LIST="8.6" python setup.py  clean --all install 
```

## Running Experiments
------------
### Download datasets
```
./download_dataset.sh
```

### Figure 7. Speedup comparison.
------------
+ **(a) Cluster GCN**. You can change the `bitwidth=4` at `0_7_eval_QGTC_cluster_GCN.py` to `[1,2,4,8]` for evaluation. default is 2 bit.
```
./0_7a_eval_QGTC_cluster_GCN.py
./1_7a_eval_DGL_cluster_GCN.py
```
Check the results in `QGTC_cluster_GCN_*bit.csv` and  `DGL_cluster_GCN.csv`. You will expect the result likes this.

| dataset          |  Epoch (ms) |
|------------------|-------------|
| artist           |  263.646    |
| soc-BlogCatalog  |  209.495    |
| ppi              |  189.016    |
| ogbn-arxiv       |  208.616    |

+ **(b) Batched GIN**. You can change the `bitwidth=4` at `0_7b_eval_QGTC_batched_GIN.py` to [1,2,4,8] for evaluation. default is 2 bit.
```
./0_7b_eval_QGTC_batched_GIN.py
./1_7b_eval_DGL_batched_GIN.py
```
Check the results in `QGTC_batched_GIN_*bit.csv` and  `DGL_batched_GIN.csv`.


## Figure 8: Additional studies.
------------
+ **(a) Comparison with the cuBLASgemmEX (int8) on Tensor Core**.
```
./3_8a_QGTC_GEMM_INT8.py
cd cuBLASGemmEX/
./compile.sh
./bench_cuBLAS_INT8.py
```
running `./2_7c_cuBLAS_INT8.py` you will get the result like this 
```
======== 1 bit ==================
X1_height 1024, X1_width: 1024, X2_width: 16, TFLOPs: 5.847
X1_height 2048, X1_width: 2048, X2_width: 16, TFLOPs: 16.605
X1_height 4096, X1_width: 4096, X2_width: 16, TFLOPs: 40.627
X1_height 1024, X1_width: 1024, X2_width: 32, TFLOPs: 11.724
X1_height 2048, X1_width: 2048, X2_width: 32, TFLOPs: 32.666
X1_height 4096, X1_width: 4096, X2_width: 32, TFLOPs: 35.032
X1_height 1024, X1_width: 1024, X2_width: 64, TFLOPs: 23.219
X1_height 2048, X1_width: 2048, X2_width: 64, TFLOPs: 37.438
X1_height 4096, X1_width: 4096, X2_width: 64, TFLOPs: 46.768
======== 2 bit ==================
X1_height 1024, X1_width: 1024, X2_width: 16, TFLOPs: 3.934
X1_height 2048, X1_width: 2048, X2_width: 16, TFLOPs: 10.086
X1_height 4096, X1_width: 4096, X2_width: 16, TFLOPs: 20.764
X1_height 1024, X1_width: 1024, X2_width: 32, TFLOPs: 7.864
X1_height 2048, X1_width: 2048, X2_width: 32, TFLOPs: 19.762
X1_height 4096, X1_width: 4096, X2_width: 32, TFLOPs: 20.951
X1_height 1024, X1_width: 1024, X2_width: 64, TFLOPs: 15.429
X1_height 2048, X1_width: 2048, X2_width: 64, TFLOPs: 25.055
X1_height 4096, X1_width: 4096, X2_width: 64, TFLOPs: 26.818
======== 4 bit ==================
X1_height 1024, X1_width: 1024, X2_width: 16, TFLOPs: 2.488
X1_height 2048, X1_width: 2048, X2_width: 16, TFLOPs: 6.561
X1_height 4096, X1_width: 4096, X2_width: 16, TFLOPs: 12.409
X1_height 1024, X1_width: 1024, X2_width: 32, TFLOPs: 4.456
X1_height 2048, X1_width: 2048, X2_width: 32, TFLOPs: 12.807
X1_height 4096, X1_width: 4096, X2_width: 32, TFLOPs: 13.929
X1_height 1024, X1_width: 1024, X2_width: 64, TFLOPs: 10.683
X1_height 2048, X1_width: 2048, X2_width: 64, TFLOPs: 12.328
X1_height 4096, X1_width: 4096, X2_width: 64, TFLOPs: 14.196
======== 8 bit ==================
X1_height 1024, X1_width: 1024, X2_width: 16, TFLOPs: 1.541
X1_height 2048, X1_width: 2048, X2_width: 16, TFLOPs: 3.483
X1_height 4096, X1_width: 4096, X2_width: 16, TFLOPs: 6.763
X1_height 1024, X1_width: 1024, X2_width: 32, TFLOPs: 3.074
X1_height 2048, X1_width: 2048, X2_width: 32, TFLOPs: 6.816
X1_height 4096, X1_width: 4096, X2_width: 32, TFLOPs: 7.366
X1_height 1024, X1_width: 1024, X2_width: 64, TFLOPs: 5.046
X1_height 2048, X1_width: 2048, X2_width: 64, TFLOPs: 6.165
X1_height 4096, X1_width: 4096, X2_width: 64, TFLOPs: 7.324
```
running `./bench_cuBLAS_INT8.py`, you will get the result like this
```
M: 1024, K: 1024, N: 16, TFLOPS: 0.55
M: 2048, K: 2048, N: 16, TFLOPS: 2.58
M: 4096, K: 4096, N: 16, TFLOPS: 3.60
M: 1024, K: 1024, N: 32, TFLOPS: 3.89
M: 2048, K: 2048, N: 32, TFLOPS: 5.49
M: 4096, K: 4096, N: 32, TFLOPS: 6.49
M: 1024, K: 1024, N: 64, TFLOPS: 4.38
M: 2048, K: 2048, N: 64, TFLOPS: 6.30
M: 4096, K: 4096, N: 64, TFLOPS: 6.65
```

+ **(b) Zero-tile jumping efficiency**.
```
./3_8a_zero_tile_jumping.py
```
**check the results in `zerotile_jumping.csv`**

+ **(c) Adjacencymatrix size impact**.
```
./3_8b_adjmatrix_size.py
```
you will get the result like this
```
X1_height 1024, X1_width: 1024, X2_width: 16, TFLOPs: 5.831
X1_height 2048, X1_width: 2048, X2_width: 16, TFLOPs: 16.323
X1_height 4096, X1_width: 4096, X2_width: 16, TFLOPs: 34.425
X1_height 1024, X1_width: 1024, X2_width: 32, TFLOPs: 11.717
X1_height 2048, X1_width: 2048, X2_width: 32, TFLOPs: 32.027
X1_height 4096, X1_width: 4096, X2_width: 32, TFLOPs: 40.175
X1_height 1024, X1_width: 1024, X2_width: 64, TFLOPs: 23.158
X1_height 2048, X1_width: 2048, X2_width: 64, TFLOPs: 37.444
X1_height 4096, X1_width: 4096, X2_width: 64, TFLOPs: 46.759
X1_height 1024, X1_width: 1024, X2_width: 128, TFLOPs: 28.417
X1_height 2048, X1_width: 2048, X2_width: 128, TFLOPs: 40.646
X1_height 4096, X1_width: 4096, X2_width: 128, TFLOPs: 52.517
X1_height 1024, X1_width: 1024, X2_width: 256, TFLOPs: 32.089
X1_height 2048, X1_width: 2048, X2_width: 256, TFLOPs: 44.151
X1_height 4096, X1_width: 4096, X2_width: 256, TFLOPs: 59.508
X1_height 1024, X1_width: 1024, X2_width: 512, TFLOPs: 41.743
X1_height 2048, X1_width: 2048, X2_width: 512, TFLOPs: 49.687
X1_height 4096, X1_width: 4096, X2_width: 512, TFLOPs: 64.172
X1_height 1024, X1_width: 1024, X2_width: 1024, TFLOPs: 37.954
X1_height 2048, X1_width: 2048, X2_width: 1024, TFLOPs: 52.970
X1_height 4096, X1_width: 4096, X2_width: 1024, TFLOPs: 66.490
```