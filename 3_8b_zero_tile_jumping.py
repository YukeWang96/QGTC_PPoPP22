#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore")

hidden = 		[16]
num_layers = 	[1]

dataset = [
        ( 'Proteins'                 , 29     , 2) ,   
		( 'artist'                 	 , 100	  , 12),
		( 'soc-BlogCatalog'	     	 , 128	  , 39),    
]

os.system("touch zerotile_jumping.log")

for n_Layer in num_layers:
	for hid in hidden:
		for data, d, c in dataset:
			command = "python cluster_gcn.py \
						--gpu 0 \
						--dataset {} \
						--dim {} \
						--n-hidden {} \
						--n-classes {} \
						--use_QGTC \
						--zerotile_jump \
						--n-epochs 1 \
						>> zerotile_jumping.log".\
						format(data, d, hid, c)		
			os.system(command)
			print()

os.system("python cluster_gcn.py --gpu 0 --dataset ppi --use_QGTC --zerotile_jump --n-epochs 1 >> zerotile_jumping.log")
print()
os.system("python cluster_gcn.py --gpu 0 --dataset ogbn-arxiv --use_QGTC --zerotile_jump --n-epochs 1 >> zerotile_jumping.log")
print()
os.system("python cluster_gcn.py --gpu 0 --dataset ogbn-products --use_QGTC --zerotile_jump --n-epochs 1 >> zerotile_jumping.log")
print()
os.system("./parse_counter.py zerotile_jumping.log > zerotile_jumping.csv")
if not os.path.exists("logs"):
	os.system("mkdir logs/")
os.system("mv *.log logs/")