# -*- coding:utf-8 -*-
# 5.1 ⼆维卷积层
import numpy as np
from numpy.core.fromnumeric import reshape
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

def main():
    """5.1.1 二维互相关运算"""
    X = np.array([[0, 1, 2, 3], 
                  [3, 4, 5, 6], 
                  [6, 7, 8, 9], 
                  [9, 10, 11, 12]])
                  
    K = np.array([[0,1],
                  [2,3]])

    Y = corr2d(X, K)
    print("Y:\n", Y)

    """5.1.2 二维卷积层"""
    X = np.ones((6, 8))
    X[:, 2:6] = 0
    K = np.array([[1, -1]]) 
    Y = corr2d(X, K)

    print("Y:\n", Y)

    """5.1.4 核数组学习"""
    # Construct a two-dimensional convolutional layer with 1 output channel and a
    # kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
    # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
    # 方法1
    conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)                     # 卷积核对象
    print("initialize kernel weight：",reshape(conv2d.weight.data, (1, 2)))     # 

    # 方法2
    # 使用自定义类
    conv2d = Conv2D(kernel_size=(1,2))
    
    # The two-dimensional convolutional layer uses four-dimensional input and
    # output in the format of (example channel, height, width), where the batch
    # size (number of examples in the batch) and the number of channels are both 1
    X = torch.tensor(X.reshape(1, 1, 6, 8), dtype=torch.float32)   # 输入
    Y = torch.tensor(Y.reshape(1, 1, 6, 7), dtype=torch.float32)   # 卷积输出

    for i in range(10):
        Y_hat = conv2d(X)
        loss = (Y_hat - Y) ** 2
        conv2d.zero_grad()      # 梯度清零
        loss.sum().backward()   # 反向传播计算梯度
        # kernel更新
        conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
        print(f'batch {i + 1}, loss {loss.sum():.3f}')

    print(torch.reshape(conv2d.weight.data,(1, 2)))

if __name__ == '__main__':
    main()