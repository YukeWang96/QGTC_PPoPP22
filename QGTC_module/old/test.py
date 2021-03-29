#!/usr/bin/env python3
import torch
import QGTC
import time

def test(M, N, K, bit_b):
    a = torch.zeros((M, K))

    # def quantize(val, bitwidth):
    #     max_val = 8
    #     min_val = 0
    #     if val > max_val:  val = max_val - 1
    #     if val < min_val:  val = min_val + 1
    #     ans = (val - min_val) * (1 << bitwidth) / (max_val - min_val)
    #     return int(ans)
    # for i in range(len(a)):
    #     for j in range(len(a[0])):
    #         tmp = quantize(a[i][j].item(), 3)
    #         print(tmp, end=" ")
    #         # print(a[i][j].data, end="")
    #     print()

    #  -- * --- reference implementation -- * ---
    # b = torch.FloatTensor([[1,1],[1,1],[1,1]])
    b = torch.zeros((K, N))

    # w = torch.ones((N, 100))
    # out = torch.mm(a, b)
    # print(out)
    # print()

    bw_a = 1
    bw_b = bit_b

    bit_a = QGTC.val2bit(a.cuda(), bw_a, False, False)
    # torch.cuda.synchronize()
    # print(" => bit encoding [a]")
    # print()
    # recover_a = QGTC.bit_recover(bit_a).cpu()
    # print(recover_a)

    # bit_b = QGTC.val2bit(b.cuda(), bw_b, True)
    bit_b = QGTC.val2bit(b.cuda(), bw_b, True, False)
    # torch.cuda.synchronize()
    # print(" => bit-encoding [b]")

    # bit_w = QGTC.val2bit(w.cuda(), bw_b, True, True)
    # torch.cuda.synchronize()
    # print(" => bit-encoding [w]")
    # print()
    # print(bit_a)
    # print(bit_b)

    num_prof = 100
    start = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(num_prof):
        int_output = QGTC.bitMM2Bit(bit_a, bit_b, M, K, N, bw_a, bw_b, bw_b)

    # print("bitMM2Bit")
    torch.cuda.synchronize()
    end = time.perf_counter()
    print("TFLOPs: {:.3f}".format(2 * M * N * K * num_prof / (end - start)/1e12))

    # float_output = QGTC.bitMM2Int(int_output, bit_w, M, N, 100, bw_a, bw_b).cpu()
    # print(float_output)
    # print("bitMM2Int")


    # bit_b = QGTC.val2bit(b.cuda(), bw_b, True, True)
    # torch.cuda.synchronize()
    # print(" => bit-encoding [b]")
    # float_output = QGTC.bitMM2Int(bit_a, bit_b, M, K, N, bw_a, bw_b).cpu()
    # print(float_output)
    # print("bitMM2Int")

    # print()
    # for i in range(len(float_output)):
    #     for j in range(len(float_output[0])):
    #         tmp = float_output[i][j].item()
    #         print(tmp, end=" ")
    #     print()


cases = [
    # [8,8,128],
    # [16,16,256],
    # [128,128,128]
    # [25200, 66, 25200]
    # [599,50,599],
    # [2058,602,2058]
    [128,128,100]
]
# scale = [1,2,3]
# example = [128,128,1024]

# for d in range(4, 11):
#     for i in range(7, 16):
#         m, k, n = (2**i), (2**i), (2**d)
#         print("m:{:6d} n:{:6d} k:{:6d}".format(m,n,k), end = " ")
#         test(m, n, k)
#     print("-------------------------")

# for j in range(2,3):
# j = 1
# test(1024,1024,1024, j)
# test(2048,2048,1024, j)
# test(4096,4096,1024, j)
# test(8192,8192,1024, j)
# print("-----------------")
# test(16384,16384,1024)

# for m, n, k in cases:
    # test(m, n, k)

def profile(path="", feat_size=1024, bitwidth=2):
    import os
    import pickle

    files = os.listdir(path)
    for fname in files:
        abs_path = os.path.join(path, fname)
        A = pickle.load(abs_path)
        N = A.size(0)
        feat = torch.ones(N, feat_size)

        bit_a = QGTC.val2bit(feat.cuda(), 1, False, False)
        bit_b = QGTC.val2bit(feat.cuda(), bitwidth, False, False)

        num_prof = 100
        start = time.perf_counter()
        torch.cuda.synchronize()
        for i in range(num_prof):
            QGTC.bitMM2Bit(bit_a, bit_b, N, N, feat_size, 1, bitwidth, bitwidth)

        # print("bitMM2Bit")
        torch.cuda.synchronize()
        end = time.perf_counter()
        print("TFLOPs: {:.3f}".format(2 * N * N * feat_size * num_prof / (end - start)/1e12))


    