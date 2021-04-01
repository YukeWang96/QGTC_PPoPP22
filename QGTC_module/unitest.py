#!/usr/bin/env python3
import torch
import QGTC
import time


def print_tensor(input_tensor):
    height = input_tensor.size(0)
    width = input_tensor.size(1)
    print("----------------------------")
    for h in range(height):
        for w in range(width):
            print(input_tensor[h][w].item(), end=" ")
        print()
    print("----------------------------")
##########################
# *test_bitencodingAndDecoding
##########################
def test_bitencodingAndDecoding(size=32, nbits=3):
    A = torch.ones((size, size)).cuda()
    height = A.size(0)
    width = A.size(1)

    # row-major, hidden.
    bit_a = QGTC.val2bit(A, nbits, False, False)
    # print(bit_a)
    # print(bit_a.size())
    A_val = QGTC.bit2val(bit_a, nbits, height, width, False, False)
    # print(A_val)

    # column-major, hidden. PAD128(width)
    bit_a = QGTC.val2bit(A, nbits, True, False)
    A_val = QGTC.bit2val(bit_a, nbits, height, width, True, False)
    # print(A_val)

    # column-major, output. PAD8(width)
    bit_a = QGTC.val2bit(A, nbits, True, True)
    print(bit_a)
    print(bit_a.size())
    A_val = QGTC.bit2val(bit_a, nbits, height, width, True, True)
    print(A_val)

##########################
# *TEST_bitMM2bit
##########################
def TEST_bitMM2bit(M=3, K=3, N=3, nbits_a=2, nbits_b=2, nbits_c=2):
    A = torch.ones((M, K)).cuda()
    B = torch.ones((K, N)).cuda()
    # print("A:\n", A)
    # print("B:\n", B)
    
    # hidden.
    bit_a = QGTC.val2bit(A, nbits_a, False, False)
    bit_b = QGTC.val2bit(B, nbits_b, True, False)

    bit_c = QGTC.bitMM2Bit(bit_a, bit_b, M, K, N, nbits_a, nbits_b, nbits_c)
    C = QGTC.bit2val(bit_c, nbits_a, M, N, False, False)
    print("C: \n", C)

##########################
# *TEST_bitMM2Int
##########################
def TEST_bitMM2Int(M=3, K=3, N=3, nbits_a=2, nbits_b=2):
    A = torch.ones((M, K)).cuda()
    B = torch.ones((K, N)).cuda()
    print("A:\n", A)
    print("B:\n", B)

    # hidden.
    bit_a = QGTC.val2bit(A, nbits_a, False, False)
    bit_b = QGTC.val2bit(B, nbits_b, True, False)

    C = QGTC.bitMM2Int(bit_a, bit_b, M, K, N, nbits_a, nbits_b)
    print("C: \n", C)
    print(C.size())

    # output layer.
    bit_a = QGTC.val2bit(A, nbits_a, False, True)
    bit_b = QGTC.val2bit(B, nbits_b, True, True)
    C = QGTC.bitMM2Int(bit_a, bit_b, M, K, N, nbits_a, nbits_b)
    print("C: \n", C)


##########################
# *TEST_GCNConv
##########################
def TEST_GCNConv(N=3, D=3, D1=3, nbits_a=2, nbits_x=2, nbits_w=2):
    X = torch.ones((N, D)).cuda()
    W = torch.ones((D, D1)).cuda()
    A = torch.ones((N, N)).cuda()

    print("A:\n", A)
    print("X:\n", X)
    print("W:\n", W)

    # hidden.
    bit_x = QGTC.val2bit(X, nbits_x, False, False)
    bit_w = QGTC.val2bit(W, nbits_w, True, False)
    bit_a = QGTC.val2bit(A, nbits_a, False, False)

    bit_XW_col = QGTC.bitMM2Bit_col(bit_x, bit_w, N, D, D1, nbits_a, nbits_x, nbits_x)
    # print(bit_XW_col)                       # column-major: nbits*PAD8(height), STEP128(width)*4}
    # print(bit_XW_col.size())
    # print("N: {}, D1: {}".format(N, D1))
    # print_tensor(bit_XW_col)

    val_XW = QGTC.bit2val(bit_XW_col, nbits_x, N, D1, True, False)
    print("**********************")
    print("1: val_AX: \n", val_XW)
    int_AXW = QGTC.bitMM2Int(bit_a, bit_XW_col, N, N, D1, nbits_a, nbits_x, True)
    print("int_AXW:\n", int_AXW)

    # bit_AXW = QGTC.bitMM2Bit(bit_a, bit_XW_col, N, N, D1, nbits_a, nbits_x, nbits_x)
    # val_AXW = QGTC.bit2val(bit_AXW, nbits_x, N, D1, False, False)
    # print("1: val_AX: \n", val_AXW)

    # ref = torch.mm(X, W)
    # int_AXW = QGTC.bitMM2Int(bit_AX, bit_w, M, K, N, nbits_a, nbits_w)
    # print("int_AXW:\n", int_AXW)
    # bit_AXW = QGTC.bitMM2Bit(bit_AX, bit_w, M, N, N1, nbits_a, nbits_x, nbits_x)
    # val_AXW= QGTC.bit2val(bit_AXW, nbits_x, M, N1, False, False)
    # print("2: val_AXW: \n", val_AXW)

##########################
# *TEST_GINConv
##########################
def TEST_GINConv(M=3, K=3, N=3, N1=3, nbits_a=2, nbits_x=2, nbits_w=2):
    A = torch.ones((M, K)).cuda()
    X = torch.ones((K, N)).cuda()
    W = torch.ones((N, N1)).cuda()

    print("A:\n", A)
    print("X:\n", X)
    print("W:\n", W)

    # hidden.
    bit_a = QGTC.val2bit(A, nbits_a, False, False)
    bit_x = QGTC.val2bit(X, nbits_x, True, False)
    bit_w = QGTC.val2bit(W, nbits_w, True, False)

    bit_AX = QGTC.bitMM2Bit(bit_a, bit_x, M, K, N, nbits_a, nbits_x, nbits_x)
    # val_AX = QGTC.bit2val(bit_AX, nbits_a, M, N, False, False)
    # print("1: val_AX: \n", val_AX)
    int_AX = QGTC.bitMM2Int(bit_a, bit_x, M, K, N, nbits_a, nbits_x)
    print("int_AX:\n", int_AX)

    int_AXW = QGTC.bitMM2Int(bit_AX, bit_w, M, K, N, nbits_x, nbits_w)
    print("int_AXW:\n", int_AXW)

    # bit_AXW = QGTC.bitMM2Bit(bit_AX, bit_w, M, N, N1, nbits_a, nbits_x, nbits_x)
    # val_AXW= QGTC.bit2val(bit_AXW, nbits_x, M, N1, False, False)
    # print("2: val_AXW: \n", val_AXW)



if __name__ == '__main__':
    # test_bitencodingAndDecoding()
    # TEST_bitMM2Int(M=3, K=3, N=3, nbits_a=3, nbits_b=3)
    # for T in range(64):
    # T = 32
    # TEST_bitMM2bit(M=T, K=T, N=T, nbits_a=2, nbits_b=2)
    # TEST_GINConv(M=3, K=3, N=3, N1=3, nbits_a=1, nbits_x=2, nbits_w=2)
    TEST_GCNConv(N=8, D=128, D1=8, nbits_a=1, nbits_x=2, nbits_w=2)