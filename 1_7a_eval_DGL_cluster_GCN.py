#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore")

hidden = 		[16]
num_layers = 	[1]
partitions = 	[1500] 

dataset = [
		( 'artist'                 	 , 100	  , 12),
		( 'soc-BlogCatalog'	     	 , 128	  , 39),    
]

os.system("touch DGL_cluster_GCN.log")

for n_Layer in num_layers:
	for hid in hidden:
		for data, d, c in dataset:
			for p in partitions:
				command = "python cluster_gcn.py --gpu 0 \
							--dataset {} \
           					--dim {} \
                            --n-hidden {} \
                            --n-classes {} \
							--psize {}\
							--regular >> DGL_cluster_GCN.log".\
							format(data, d, hid, c, p)		
				os.system(command)
				print()

os.system("python cluster_gcn.py --gpu 0 --dataset ppi --regular >> DGL_cluster_GCN.log")
print()
os.system("python cluster_gcn.py --gpu 0 --dataset ogbn-arxiv --regular >> DGL_cluster_GCN.log")
print()
# os.system("python cluster_gcn.py --gpu 0 --dataset ogbn-products --regular")

os.system("./parse_time.py DGL_cluster_GCN.log > DGL_cluster_GCN.csv")
if not os.path.exists("logs"):
	os.system("mkdir logs/")
os.system("mv *.log logs/")