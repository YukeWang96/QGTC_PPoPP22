#!/usr/bin/env python3
import os


hidden = [128] #[16, 32, 64, 128, 256]
num_layers = [1]
data_dir = '/home/yuke/.graphs/orig/'
partitions = [1500] # 1500, 3000, 4500, 6000, 7500, 9000]

dataset = [
		# ('toy'	        , 3	    , 2   ),  
		# ('tc_gnn_verify'	, 16	, 2),
		# ('tc_gnn_verify_2x'	, 16	, 2),

		# ('citeseer'	        	, 3703	    , 6   ),  
		# ('cora' 	        		, 1433	    , 7   ),  
		# ('pubmed'	        		, 500	    , 3   ),      
		# ('ppi'	            	, 50	    , 121 ),   
		
		# ('OVCAR-8H'                  , 66       , 2) , 
		# ('Yeast'                     , 74       , 2) ,
		# ('DD'                        , 89       , 2) ,
		# ('SW-620H'                   , 66       , 2) ,


		('PROTEINS_full'             , 29     , 2) ,   
		# ( 'artist'                 	 , 100	  , 12),
		# ( 'soc-BlogCatalog'	     	 , 128	  , 39),      



		# ( 'amazon0505'               , 96	  , 22),
		# ( 'com-amazon'             , 96	  , 22),
		# ( 'amazon0601'  	         , 96	  , 22), 
    	# ( 'web-BerkStan'           , 100	  , 12),

		# ('YeastH'                    , 75       , 2) ,   
		# ( 'wiki-topcats'             , 300	  , 12),

	    # ( 'reddit'                   , 602    , 41),
		# ( 'COLLAB'                   , 100      , 3) ,
		# ( 'Reddit'                   , 602      , 41),
		# ( 'enwiki-2013'	           , 100	  , 12),      
		# ( 'amazon_also_bought'       , 96       , 22),
]

for n_Layer in num_layers:
	for hid in hidden:
		for data, d, c in dataset:
			print("=> {}, hiddn: {}".format(data, hid))
			for p in partitions:
				command = "python cluster_gcn.py --gpu 0 --dataset {} --dim {} --n-classes {} --psize {}".\
							format(data, d, c, p)		
				# command = "sudo ncu --csv --set full python main_gcn.py --dataset {0} --dim {1} --hidden {2} --classes {3} --num_layers {4} --model {5} | tee prof_{0}.csv".format(data, d, hid, c, n_Layer, model)		
				os.system(command)
				print()
		print("----------------------------")
	print("===========================")