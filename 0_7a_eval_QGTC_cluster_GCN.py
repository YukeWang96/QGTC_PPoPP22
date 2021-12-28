#!/usr/bin/env python3
import os
import warnings
warnings.filterwarnings("ignore")

hidden = 		[16] 
num_layers = 	[1]
partitions = 	[1500]

<<<<<<< HEAD
bitwidth = 2
=======
bitwidth = 2 # 1,2,4,8
>>>>>>> 69486b42730fe849b8dcff4f144caf02deb03773

dataset = [
        ( 'Proteins'             	 , 29     , 2) ,   
		( 'artist'                 	 , 100	  , 12),
		( 'soc-BlogCatalog'	     	 , 128	  , 39),    
]

os.system("touch res_QGTC_cluster_GCN_{}bit.log".format(bitwidth))

for n_Layer in num_layers:
	for hid in hidden:
		for data, d, c in dataset:
			for p in partitions:
				command = "python cluster_gcn_{}.py \
    						--gpu 0 \
							--dataset {} \
       						--dim {} \
                            --n-hidden {} \
                            --n-classes {} \
							--psize {}\
							--use_QGTC \
           					>> res_QGTC_cluster_GCN_{}bit.log".\
							format(bitwidth, data, d, hid, c, p, bitwidth)		
				os.system(command)
				print()

os.system("python cluster_gcn_{0}.py --gpu 0 --dataset ppi --use_QGTC >> res_QGTC_cluster_GCN_{0}bit.log".format(bitwidth))
print()
os.system("python cluster_gcn_{0}.py --gpu 0 --dataset ogbn-arxiv --use_QGTC >> res_QGTC_cluster_GCN_{0}bit.log".format(bitwidth))
print()
os.system("python cluster_gcn_{0}.py --gpu 0 --dataset ogbn-products --use_QGTC >> res_QGTC_cluster_GCN_{0}bit.log".format(bitwidth))
print()
os.system("./parse_time.py res_QGTC_cluster_GCN_{0}bit.log > res_QGTC_cluster_GCN_{0}bit.csv".format(bitwidth))
if not os.path.exists("logs"):
	os.system("mkdir logs/")
os.system("mv *.log logs/")