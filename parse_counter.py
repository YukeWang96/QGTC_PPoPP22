#!/usr/bin/env python3
import os
import sys

fp = open(sys.argv[1])

global_counter = []
counter = []
data_li = []
for line in fp:
    if 'dataset' in line:
        data = line.split(",")[1].split("=")[1].strip('\'')
        # print(global_counter)
        # print(counter)
        data_li.append([data, sum(global_counter), sum(counter), (sum(global_counter) - sum(counter))/sum(global_counter)])
        global_counter = []
        counter = []
    if 'counter_global:' in line:
        data = int(line.split(":")[1].strip('\''))
        global_counter.append(data)
        continue
    if "counter:" in line:
        data = int(line.split(":")[1].strip('\''))
        counter.append(data)

print("dataset",",", "non-jumping",",","jumping",",","reduction ratio")
for dat, gc, c, ratio in data_li:
    print(dat,",", gc,",",c,",","{:.3f}".format(ratio))