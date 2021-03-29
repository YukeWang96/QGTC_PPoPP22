#!/usr/bin/env python3
import torch
import QGTC
import time

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

def bitMM2bit():
    pass

def bitMM2Int():
    pass

if __name__ == '__main__':
    test_bitencodingAndDecoding()