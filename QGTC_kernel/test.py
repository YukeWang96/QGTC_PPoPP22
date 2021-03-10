#!/usr/bin/env python3
import torch
import sys
import QGTC

# a = torch.FloatTensor([[1,2,3],[1,2,3]])
a = torch.ones((8, 128))
# print(a)

def quantize(val, bitwidth):
    max_val = 8
    min_val = 0
    if val > max_val:  val = max_val - 1
    if val < min_val:  val = min_val + 1
    ans = (val - min_val) * (1 << bitwidth) / (max_val - min_val)
    return int(ans)

# for i in range(len(a)):
#     for j in range(len(a[0])):
#         tmp = quantize(a[i][j].item(), 3)
#         print(tmp, end=" ")
#         # print(a[i][j].data, end="")
#     print()

#  -- * --- reference implementation -- * ---
# b = torch.FloatTensor([[1,1],[1,1],[1,1]])
b = torch.ones((128, 8))
out = torch.mm(a, b)
# print(out)
# print()

bw_a = 3
bw_b = 3

bit_a = QGTC.bit_qnt(a.cuda(), bw_a, False)
torch.cuda.synchronize()
print(" => bit encoding [a]")
print()
# recover_a = QGTC.bit_recover(bit_a).cpu()
# print(recover_a)

bit_b = QGTC.bit_qnt(b.cuda(), bw_b, True)
torch.cuda.synchronize()
print(" => bit-encoding [b]")
print()

# print(bit_a)
print(bit_b)
print(bit_b.shape)

# int_output = QGTC.mm_v1(bit_a, bit_b, 8, 128, 8, bw_a, bw_b, bw_a).cpu()
# print("mm_v1")
# print()

# float_output = QGTC.mm_v2(bit_a, bit_b, 8, 128, 8, bw_a, bw_b).cpu()
# print(float_output)
# print("mm_v2")
# print()
# for i in range(len(float_output)):
#     for j in range(len(float_output[0])):
#         tmp = float_output[i][j].item()
#         print(tmp, end=" ")
#     print()