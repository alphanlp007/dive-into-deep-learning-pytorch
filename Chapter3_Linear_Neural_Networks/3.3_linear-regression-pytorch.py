# -*- coding:utf-8 -*-
# 2021.04.11

import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
from torch.nn import init

torch.manual_seed(1)
torch.set_default_tensor_type('torch.FloatTensor')

# 线性回归模型定义
class LinearRegressionNet(nn.Module):
    def __init__(self, n_features):
        super(LinearRegressionNet, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        y = self.linear(x)
        return y

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
    # params config
    num_examples = 1000
    num_inputs = 2
    batch_size = 10

    # 模拟数据集生成
    features, labels = generateData(num_examples, num_inputs)

    # 将训练数据与标签数据组合
    dataset = Data.TensorDataset(features, labels)

    # 将dataset放入DataLoader中
    data_iter = Data.DataLoader(
                                dataset=dataset,        # torch TensorDataset格式
                                batch_size=batch_size,  # mini batch size
                                shuffle=True,           # 随机打乱训练数据
                                num_workers=2,          # 多线程读取数据
    )
    for X, y in data_iter:
        print(X,'\n',y)
        break

    # 模型定义
    # 网络定义方法1
    net = LinearRegressionNet(num_inputs)

    # # 网络定义方法2
    # net = nn.Sequential(nn.Linear(num_inputs, 1)
    #                     # 此处还可以传入其他层
    # )

    # # 网络定义方法3
    # net = nn.Sequential()
    # net.add_module('linear', nn.Linear(num_inputs, 1))
    # # net.add_module ......

    # # 网络定义方法4
    # from collections import OrderedDict
    # net = nn.Sequential(OrderedDict([
    #                                 ('linear', nn.Linear(num_inputs, 1))
    #                                 # ......
    #                                 ])
    #         )

    print("Neural network structure:", net)

    # 网络参数输出
    for param in net.parameters():
        print(param)
        print(param.data)

    for m in net.modules():
        print(m[0])
        m.weight.data.normal_(0, 0.01)
        m.weight.data.normal_(0, 0.02)
        print(m.weight.data)
        print(m.bias.data)


    # # 网络参数初始化
    # init.normal_(net.weight, mean=0, std=0.01)
    # init.constant_(net.bias,val=0.0)
    



if __name__ == '__main__':
    main()