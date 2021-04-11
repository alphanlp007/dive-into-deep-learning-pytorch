# -*- coding:utf-8 -*-
# 2021.04.10

import torch
import numpy as np
from matplotlib import pyplot as plt

# 导入当前目录的父目录的package
import os,sys
sys.path.append("..")
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils.utils import set_figsize, data_iter, linear_regression, squared_loss, sgd

def generateData(nums_samples, nums_features):
    """
        生成模拟数据集 y = X*w + b
        返回Tensor类型
    """

    # config weights and bias
    true_w = [2, -3.4]
    true_b = 4.2

    # 标准正态分布生成特征，均值为0，方差为1
    features = torch.randn(nums_samples, nums_features, dtype=torch.float32)
    labels   = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b

    # 添加均值为0，方差为0.01的噪声项
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                           dtype=torch.float32)
    
    return features, labels

def main():
    # check torch version
    print("pytorch version:", torch.__version__)

    # 模拟数据生成
    num_inputs      = 2
    num_examples    = 1000
    features, labels = generateData(num_examples, num_inputs)

    # 特征标签图
    set_figsize()
    plt.scatter(features[:,1].numpy(), labels.numpy(), s=1)
    plt.savefig('./figures/3.1_linear-regression-scratch.jpg')

    # 读取数据
    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break

    # 初始化权重参数和偏置参数, 权重参数服从均值为0，方差为0.01的正态分布
    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    # 模型训练
    lr = 0.03
    num_epochs = 10
    net = linear_regression
    loss = squared_loss
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y).sum() # 损失函数输出为标量
            l.backward()                    # 反向传播计算梯度
            sgd([w, b], lr, batch_size)     # 使用小批量随机梯度下降迭代模型参数
            
            # 不要忘了梯度清零，pytorch没有梯度自动清零的功能，手动清零可以避免梯度累加
            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

    print("true_w = [2, -3.4]", '\n', w)
    print("true_b = 4.2", '\n', b)

if __name__ == '__main__':
    print('current path:',os.path.abspath(__file__))
    main()