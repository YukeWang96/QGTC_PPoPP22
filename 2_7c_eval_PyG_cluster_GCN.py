#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore")

hidden = 		[16] 
num_layers = 	[1]
partitions = 	[1500] 

dataset = [
        ( 'Proteins'             	 , 29     , 2),   
		( 'artist'                 	 , 100	  , 12),
		( 'soc-BlogCatalog'	     	 , 128	  , 39),    
]

os.system("touch PyG_cluster_GCN.log")

for n_Layer in num_layers:
	for hid in hidden:
		for data, d, c in dataset:
			print("=> {}, hiddn: {}".format(data, hid))
			for p in partitions:
				command = "python cluster_gcn_pyg.py \
        					--gpu 0 \
							--dataset {} \
           					--dim {} \
                            --n-hidden {} \
                            --n-classes {} \
							--psize {}\
							--regular \
							--use_PyG >> PyG_cluster_GCN.log".\
							format(data, d, c, hid, p)		
				os.system(command)
				print()
 

os.system("python cluster_gcn_pyg.py --gpu 0 --dataset ppi --regular --use_PyG >> PyG_cluster_GCN.log")
print()
os.system("python cluster_gcn_pyg.py --gpu 0 --dataset ogbn-arxiv --regular --use_PyG >> PyG_cluster_GCN.log")
print()
os.system("python cluster_gcn_pyg.py --gpu 0 --dataset ogbn-products --regular --use_PyG >> PyG_cluster_GCN.log")
print()
os.system("./parse_time.py PyG_cluster_GCN.log > PyG_cluster_GCN.csv")
if not os.path.exists("logs"):
	os.system("mkdir logs/")
os.system("mv *.log logs/")