# -*- coding:utf-8 -*-
# 5.2 填充和步幅
import numpy as np
import torch
from torch import nn

# 二维卷积类定义
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size)) # 随机初始化卷积核
        self.bias   = nn.Parameter(torch.zeros(1))          # 偏移量

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias           # 卷积计算结果

def corr2d(X, kernel):
    """
    功能：
        二维卷积互相关运算，Compute 2D cross-correlation.
    Input:
        X: 输入数组
        kernel: 核数组
    卷积计算公式(W, H):分别表示图像的宽度和高度
        (W - F + 2*P)/stride + 1
        (H - F + 2*P)/stride + 1
    """
    # kernel处理和卷积结果初始化
    if isinstance(X, np.ndarray) or isinstance(kernel, np.ndarray):
        X = torch.tensor(X)
        kernel = torch.tensor(kernel)

    h, w = kernel.shape # [rows，cols]
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))

    # 互相关运算，以下运算对单通道矩阵输入有效
    for i in range(Y.shape[0]): # row
        for j in range(Y.shape[1]): # col
            Y[i, j] = (X[i : i + h, j : j + w] * kernel).sum()

    return Y

def comp_conv2d(conv2d, X):
    """二维卷积计算"""
    X = X.reshape((1, 1) + X.shape) # (1, 1) + (8, 8) = (1, 1, 8, 8)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


def main():
    # 1. 定义卷积核3*3, padding=1,stride=1
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    print(conv2d.weight.data)
    X = torch.rand(size=(8, 8)) # H/W: (8-3+2*1)/1 + 1 = 8
    print(comp_conv2d(conv2d, X).shape) # torch.Size([8, 8])

    # 2. 定义卷积核5*3，padding=(2,1),stride=1
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2,1))
    X = torch.rand(size=(8, 8)) # H: (8-5+2*2)/1 + 1 = 8, W: (8-3+2*1)/1 + 1=8
    print(comp_conv2d(conv2d, X).shape) # torch.Size([8, 8])
    
    # 3. 定义卷积核3*3, padding=1, stride=2
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    X = torch.rand(size=(8, 8)) # H: (8-3+2*1)/2 + 1 = 4, W: (8-3+2*1)/2 + 1=4 (取整计算)
    print(comp_conv2d(conv2d, X).shape) # torch.Size([4, 4])

    # 4. 定义卷积核3*3, padding=1, stride=2
    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(1, 3))
    X = torch.rand(size=(8, 8)) # H: (8-3+2*0)/1 + 1 = 6, W: (8-5+2*1)/1 + 1 = 2
    print(comp_conv2d(conv2d, X).shape) # torch.Size([6, 2])


if __name__ == '__main__':
    main()