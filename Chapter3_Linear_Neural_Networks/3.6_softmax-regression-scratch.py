# -*- coding:utf-8 -*-
# 2021.04.12
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 导入当前目录的父目录的package
import os,sys
sys.path.append("..")
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils.utils import show_fashion_mnist, get_fashion_mnist_labels, load_data_fashion_mnist

def net(X, W, b):
    """
    单层神经softmax神经网络，有多个分类输出
    X：the input tensor, size is [batch_size, num_inputs]
    W: the weight tensor, size is [num_inputs, class]
    b: the bias tensor, value is constant zero
    """
    _, num_inputs = X.shape
    y = torch.mm(X.view(-1, num_inputs), W) + b
    output = softmax(y)
    return output

def softmax(X):
    """
    X : the input tensor, size is [batch_size, num_inputs]
    """
    X_exp = X.exp()
    partition = X_exp.sum(dim = 1, keepdim = True) # dim=0,纵向求和，dim=1,横向求和
    
    # 广播机制
    return X_exp / partition

def cross_entropy(y_hat, y):
    """交叉熵衡量两个概率分布的相关性"""
    """calculate single sample cross entropy"""
    # unit test
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    y = torch.tensor([0, 2])
    print(y_hat.view(-1,3))
    print(y.view(-1,1))

    # torch.gather(input, dim, index, out=None) 
    # 沿给定轴dim，提取index指定位置的值, y_hat为softmax输出的概率分布[0.1 0.02 0.3 ... 0.5]
    # 由于预测的多个类别概率分布中只有一个类别是正确的，单个样本的损失函数loss = -log(y_hat_yi)
    loss = torch.gather(y_hat, 1, y.view(-1, 1))
    return -torch.log(loss) # 计算batch中每个样本的交叉熵损失

def main():
    # Step 1：config parameters
    batch_size = 256
    num_inputs = 784
    num_outputs = 10

    num_epochs = 30
    lr = 0.1

    # Step 2：加载训练数据和测试数据迭代对象
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    # 数据集测试
    from matplotlib import pyplot as plt
    for X, y in train_iter:
        print('Image:',X.shape,'label:',y.shape)
        img = X[0].view(28,28).numpy()
        label = get_fashion_mnist_labels([y[0]])

        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
        plt.savefig('../figures/3.6_train_iter_test.jpg')

        break

    cross_entropy(0, 0)

if __name__ == '__main__':
    main()