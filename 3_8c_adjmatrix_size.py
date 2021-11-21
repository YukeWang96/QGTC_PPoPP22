#!/usr/bin/env python3
import torch
import QGTC

def PROFILE_NonZeroTile(M=3, K=3, N=3, nbits_a=1, nbits_x=1):
    A = torch.ones((M, K)).cuda()
    X = torch.ones((K, N)).cuda()

    bit_a = QGTC.val2bit(A, nbits_a, False, False)
    bit_x = QGTC.val2bit(X, nbits_x, True, False)

    QGTC.bitMM2Bit_profile(bit_a, bit_x, M, K, N, nbits_a, nbits_x, nbits_x)

if __name__ == '__main__':
	for dim in [16, 32, 64, 128, 256, 512, 1024]:
		for T in range(3):
			N = (2**T) * 1024
			PROFILE_NonZeroTile(N, N, dim, nbits_x=1)