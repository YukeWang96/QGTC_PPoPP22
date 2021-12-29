#!/usr/bin/env python3
import os
import sys

fp = open(sys.argv[1])

item_li = []
data_li = []
print("dataset", ",","Epoch (ms)")
for line in fp:
    if 'dataset' in line:
        if len(line.split(",")) < 5: continue
        data = line.split(",")[2].split("=")[1].strip('\'')
        data_li.append(data)
    if "Avg. Epoch:" in line:
        ti = float(line.strip("\n").split(":")[1].strip("ms"))
        item_li.append(ti)
        
for dat, res in zip(data_li, item_li):
    print(dat, ",", res)