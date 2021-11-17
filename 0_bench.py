#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore")

hidden = [16] #[16, 32, 64, 128, 256]
num_layers = [1]
data_dir = 'graphs/'
partitions = [1500] # 1500, 3000, 4500, 6000, 7500, 9000]

dataset = [
		('ppi'	            		 , 50	  , 121),   
		( 'artist'                 	 , 100	  , 12),
		( 'soc-BlogCatalog'	     	 , 128	  , 39),    
]

for n_Layer in num_layers:
	for hid in hidden:
		for data, d, c in dataset:
			print("=> {}, hiddn: {}".format(data, hid))
			for p in partitions:
				command = "python cluster_gcn.py --gpu 0 \
							--dataset {} --dim {} --n-hidden {} --n-classes {} \
							--psize {}\
							--use_QGTC".\
							format(data, d, c, hid, p)		
				# command = "sudo ncu --csv --set full python main_gcn.py --dataset {0} --dim {1} --hidden {2} --classes {3} --num_layers {4} --model {5} | tee prof_{0}.csv".format(data, d, hid, c, n_Layer, model)		
				os.system(command)
				print()
		print("----------------------------")
	print("===========================")