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

#  -- * --- reference implementation -- * ---
b = torch.FloatTensor([[1,1],[2,1],[3,1]])
out = torch.mm(a, b)
print(out)

bit_a = QGTC.bit_qnt(a.cuda(), 3, False)
torch.cuda.synchronize()
print(" => bit encoding [a]")
print()

bit_b = QGTC.bit_qnt(b.cuda(), 3, True)
torch.cuda.synchronize()
print(" => bit-encoding [b]")
print()

float_output = QGTC.mm_v1(bit_a, bit_b, 2, 3, 1, 3, 3, 3).cpu()
print("mm_v1")
print()

float_output = QGTC.mm_v2(bit_a, bit_b, 2, 3, 1, 3, 3).cpu()
print("mm_v2")
print()

# for i in range(len(ans)):
#     for j in range(len(ans[0])):
#         tmp =  quantize(a[i][j].item(), 3)
#         print(tmp, end=" ")
#         # print(a[i][j].data, end="")
#     print()
