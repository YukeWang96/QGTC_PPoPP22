#!/usr/bin/env python3
import os

if __name__ == '__main__':
    for dim in [16, 32, 64]:
        for T in range(3):
            N = (2**T) * 1024
            os.system("./cublas_main {} {} {}".format(N,N,dim))