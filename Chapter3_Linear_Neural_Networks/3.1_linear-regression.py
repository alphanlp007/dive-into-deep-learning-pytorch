# -*- coding:utf-8 -*-
# 2021.04.10

import torch
from time import time

def main():
    # check torch version
    print("pytorch version:", torch.__version__)

    # solution 1
    a = torch.ones(1000)
    b = torch.ones(1000)
    start = time()
    c = torch.zeros(1000)
    for i in range(1000):
        c[i] = a[i] + b[i]
    print(time() - start)

    # solution 2
    start = time()
    d = a + b
    print(time() - start)

    # 广播机制
    a = torch.ones(3)
    b = 10
    print(a + b)

if __name__ == '__main__':
    main()