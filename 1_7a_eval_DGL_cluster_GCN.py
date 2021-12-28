#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore")

hidden = 		[16]
num_layers = 	[1]
partitions = 	[1500] 

os.system("touch res_DGL_cluster_GCN.log")

dataset = [
    	('Proteins'             	 , 29     , 2) ,   
		( 'artist'                 	 , 100	  , 12),
		( 'soc-BlogCatalog'	     	 , 128	  , 39),    
]

for n_Layer in num_layers:
	for hid in hidden:
		for data, d, c in dataset:
			for p in partitions:
				command = "python cluster_gcn_dgl.py \
        					--gpu 0 \
							--dataset {} \
           					--dim {} \
                            --n-hidden {} \
                            --n-classes {} \
							--psize {}\
							--regular >> res_DGL_cluster_GCN.log".\
							format(data, d, hid, c, p)		
				os.system(command)
				print()

os.system("python cluster_gcn_dgl.py --gpu 0 --dataset ppi --regular >> res_DGL_cluster_GCN.log")
print()
os.system("python cluster_gcn_dgl.py --gpu 0 --dataset ogbn-arxiv --regular >> res_DGL_cluster_GCN.log")
print()
os.system("python cluster_gcn_dgl.py --gpu 0 --dataset ogbn-products --regular >> res_DGL_cluster_GCN.log")
print()
os.system("./parse_time.py res_DGL_cluster_GCN.log > res_DGL_cluster_GCN.csv")
if not os.path.exists("logs"):
	os.system("mkdir logs/")
os.system("mv *.log logs/")