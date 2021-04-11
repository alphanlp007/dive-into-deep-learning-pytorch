# -*- coding:utf-8 -*-
# 2021.04.11

import torch
from torch import nn
from torch.nn import init
import torch.optim as optim
import torch.utils.data as Data
import numpy as np

# 线性回归模型定义
class LinearRegressionNet(nn.Module):
    def __init__(self, n_features,n_outputs = 1):
        super(LinearRegressionNet, self).__init__()
        self.linear = nn.Linear(n_features, n_outputs)

    def forward(self, x):
        """前向传播"""
        y = self.linear(x)
        return y

def generateData(nums_samples, nums_features):
    """
        生成模拟数据集 y = X*w + b
        返回Tensor类型
    """

    # config true weights and bias
    true_w = [2, -3.4]
    true_b = 4.2

    # 标准正态分布生成特征，均值为0，方差为1
    features = torch.randn(nums_samples, nums_features, dtype=torch.float32)
    labels   = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b

    # 添加均值为0，方差为0.01的噪声项
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
    
    return features, labels

def data_iter(batch_size, features, labels):
    # 组合训练数据与标签数据
    dataset = Data.TensorDataset(features, labels)

    # dataset放入DataLoader中
    data_iter = Data.DataLoader(
                                dataset=dataset,        # torch TensorDataset格式
                                batch_size=batch_size,  # mini batch size
                                shuffle=True,           # 随机打乱训练数据
                                num_workers=2,          # 多线程读取数据
    )

    return data_iter

def train(features, labels):
    ### Step 1：模型结构搭建 ###
    num_inputs = 2

    # 网络结构定义方法1(单层神经网络，不能对层进行索引)
    net = LinearRegressionNet(num_inputs)
    
    # # # 以下几个网络定义方法可以按层索引 net[0],net[1] # # #
    # # 网络结构定义方法2
    # net = nn.Sequential(LinearRegressionNet(num_inputs, 1)
    #                     # 此处还可以传入其他层
    # )

    # # 网络结构定义方法3
    # net = nn.Sequential()
    # net.add_module('linear', LinearRegressionNet(num_inputs, 1))
    # # net.add_module ......

    # # 网络结构定义方法4
    # from collections import OrderedDict
    # net = nn.Sequential(OrderedDict([
    #                                 ('linear', LinearRegressionNet(num_inputs, 1))
    #                                 # ......
    #                                 ])
    # )

    # 输出网络结构和所有的网络参数，每层神经网络的参数包括权重参数weight和偏置参数bias
    print("Neural network structure:", net)
    # print("net[0]:", net[0])
    # print(net.parameters())             # <generator object>
    # print(list(net.parameters()))       # <list object>
    # print(list(net.parameters())[0])
    # print(list(net.parameters())[1])
    # for param in net.parameters():
    #     print(param)

    ### Step 2：初始化模型参数，模型参数的初始化有很多种 ###
    weight_0 = list(net.parameters())[0]
    bias_0   = list(net.parameters())[1]
    init.normal_(weight_0, mean=0,std=0.01)
    init.constant_(bias_0,val=0)
    print("init weight_0:{}, bias_0:{}".format(weight_0, bias_0))

    ### Step 3：定义损失函数 ###
    loss = nn.MSELoss()

    ### Step 4：定义优化算法，优化算法有很多种，如AdaGrad, RMSProp, AdaDelta, Adam ###
    optimizer = optim.SGD(net.parameters(), lr=0.03)
    print(optimizer)

    # # 不同的层使用不同的学习率
    # optimizer = optim.SGD([
    #     # 如果对某个参数不指定学习率，就使用最外层的默认学习率
    #     {'params':net[0].parameters()}, # lr = 0.03
    #     {'params':net[1].parameters(), 'lr':0.01}
    # ], lr = 0.03)
    # print(optimizer)

    ### Step 5：模型训练 ###
    num_epoch = 20
    batch_size = 10
    for epoch in range(1, num_epoch + 1):
        for X, y in data_iter(batch_size, features, labels):
            output = net(X)
            l = loss(output, y.view(-1,1))
            # optimizer.zero_grad() # 梯度清零，避免梯度累加
            l.backward()            # 计算梯度
            optimizer.step()        # 参数更新
            optimizer.zero_grad()   # 梯度清零
        print('epoch %d, loss: %f' % (epoch, l.item()))
    print("weight_0:",list(net.parameters())[0].data)
    print("bias_0:", list(net.parameters())[1].data)

def main():
    # initial configure
    torch.manual_seed(1)
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Step 1：生成数据集
    num_examples = 1000
    num_inputs = 2
    batch_size = 10

    # Step 2：模拟数据集生成
    features, labels = generateData(num_examples, num_inputs)

    # Step 3：测试获取mini batch数据
    mini_batch = data_iter(batch_size, features, labels)
    for X, y in mini_batch:
        print(X,'\n',y)
        break

    # Step 4：模型训练
    train(features, labels)

if __name__ == '__main__':
    main()