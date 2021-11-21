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

for n_Layer in num_layers:
	for hid in hidden:
		for data, d, c in dataset:
			# print("=> {}, hiddn: {}".format(data, hid))
			for p in partitions:
				command = "python cluster_gcn.py --gpu 0 \
							--dataset {} \
           					--dim {} \
                            --n-hidden {} \
                            --n-classes {} \
							--psize {}\
							--regular \
							--run_GIN".\
							format(data, d, hid, c, p)		
				os.system(command)
				print()

os.system("python cluster_gcn.py --gpu 0 --dataset ppi --regular --run_GIN")
print()
os.system("python cluster_gcn.py --gpu 0 --dataset ogbn-arxiv --regular --run_GIN")
print()
# os.system("python cluster_gcn.py --gpu 0 --dataset ogbn-products --regular")