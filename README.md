QGTC: Accelerating Quantized GNN via GPU Tensor Core
============
**Note that this project is still under active development and subject to changes.**

## Clone this project.
------------

```
git clone git@gitlab.com:YK-Wang96/ppopp22_qgtc.git
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

+ Install [**`Pytorch-Geometric (PyG)`**](https://github.com/rusty1s/pytorch_geometric).
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric
```

### Install QGTC. Go to `QGTC_module/`, then run 
```
TORCH_CUDA_ARCH_LIST="8.6" python setup.py  clean --all install 
```

## Running Experiments
------------
+ Get dataset `wget https://project-datasets.s3.us-west-2.amazonaws.com/qgtc_graphs.tar.gz` and `tar -zxvf qgtc_graphs.tar.gz`.
+ `./bench.py` for running `proteins`, `artist` and `soc-Blogcatalog` dataset.
+ `./run_ppi.sh` for running `PPI` dataset.
+ `./run_ogb.sh` for running `ogbn-arxiv` and `ogbn-products` dataset.
+ `./run_all` for running all the above three experiments together by following the order of `bench.py`, `run_ppi.sh` and `run_ogb.sh`
