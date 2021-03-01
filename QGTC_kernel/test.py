#!/usr/bin/env python3
import torch
import sys
import QGTC

a = torch.FloatTensor([[1,2,3],[4,5,6]])
print(a)

def quantize(val, bitwidth):
    max_val = 10
    min_val = -10
    if val > max_val:  val = max_val - 1
    if val < min_val:  val = min_val + 1
    ans = (val - min_val) * (1 << bitwidth) / (max_val - min_val)
    return int(ans)

for i in range(len(a)):
    for j in range(len(a[0])):
        tmp =  quantize(a[i][j].item(), 3)
        print(tmp, end=" ")
        # print(a[i][j].data, end="")
    print()
# ans = QGTC.bit_qnt(a, 3)