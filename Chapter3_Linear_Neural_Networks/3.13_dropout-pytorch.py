# -*- coding:utf-8 -*-
# 2021.04.24
import torch
from torch import nn
from torch.nn import init
import numpy as np

# 导入当前目录的父目录的package
import os,sys

from torch.nn.modules.activation import Softmax
sys.path.append("..")
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils.utils import show_fashion_mnist, get_fashion_mnist_labels, load_data_fashion_mnist

# 输入层，拉平图像
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x): 
        # [256,1,28,28] ==> [256,786]
        y = x.view(x.shape[0], -1)
        return y


def dropout(X, drop_prob):
    """
    X: the input tensor, size is [batch_size, n_dimension]
    drop_prob: the probability that the neuron output is set to zero.
    """
    X = X.float() # dtype转换为torch.float32类型变量
    assert 0 <= drop_prob <= 1

    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(X) # 生成与X形状相同、元素全为0的张量
    
    # 生成mask矩阵
    # torch.rand:返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。
    # 当keep_prob = 1.0 时，mask为全1矩阵
    # 当keep_prob = 0.5 时，mask矩阵部分为0，部分为1
    # keep_prob的值设定的越小，则mask矩阵中为0的元素就越多，则意味着被dropout的神经元也越多
    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X / keep_prob # 与教程上的计算公式保持一致，在训练阶段就除以1-p


def sgd(params, lr, batch_size):
    # 定义随机梯度下降优化算法
    for param in params:
        # 1）参数使用param.data，获取纯数据
        # 2) 梯度为何要处理batch_size，梯度是mini batch上梯度的累加，所以要除以batch_size的大小
        param.data -= lr * param.grad / batch_size


def train(net, train_iter, test_iter, loss, num_epochs, batch_size, params = None, lr = None, optimizer = None):
    """
    net：网络结构
    train_iter：训练数据迭代器
    test_iter ：测试数据集迭代
    loss：损失函数
    params：网络中待优化参数
    """
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            loss_ = loss(y_hat, y).sum() # 当前batch的交叉熵总和，即损失值，也可以计算平均损失，教程上一般写为平均损失

            ### 梯度清零，否则会出现梯度累加，再进行反向传播计算网络参数的梯度 ###
            if optimizer is not None: 
                # 使用Pytorch优化器，判断是否指定优化算法
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None: 
                # 使用自己编写的优化器，判断输入参数和梯度是否为None
                for param in params:
                    param.grad.data.zero_()

            # 损失函数反向传播
            loss_.backward()

            # 参数更新
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_loss_sum += loss_.item() # 累加每个batch的损失函数值
            train_acc_sum  += (y_hat.argmax(dim=1) == y).sum().item() # 累加每个batch预测正确的样本数量
            n += y.shape[0]

        # 单个epoch训练结束，评估模型在测试集上的精度
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss：%.4f, train_acc %.4f, test_acc %.4f"\
            % (epoch+1, train_loss_sum/n, train_acc_sum/n, test_acc))


def evaluate_accuracy(data_iter, net):
    """模型在测试集上的评估"""
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train() # 改回训练模式
        else: # 自定义的模型, func.__code__.co_varnames:将函数局部变量以元组的形式返回。
            if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
        n += y.shape[0]
    return acc_sum / n


def predict(net, test_iter):
    """模型预测"""
    X, y = iter(test_iter).next()
    print("X shape:",X.shape)
    print("y shape:",y.shape)
    if isinstance(net, torch.nn.Module):
        net.eval()      # 评估模式，这会关闭dropout
        y_hat = net(X).argmax(dim = 1)
        net.train()
    else: # 自定义的模型, func.__code__.co_varnames:将函数局部变量以元组的形式返回
        if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
            # 将is_training设置成False
            y_hat = net(X, is_training=False).argmax(dim=1)
        else:
            y_hat = net(X).argmax(dim=1)
      
    true_labels = get_fashion_mnist_labels(y.numpy())
    predict_labels = get_fashion_mnist_labels(y_hat.numpy())

    titles = [true + '\n' + pred for true, pred in zip(true_labels, predict_labels)]
    show_fashion_mnist(X[10:19], titles[10:19], imagepath = '../figures/3.13_dropout-pytorch.jpg')


def main():
    """begin 神经网络参数配置"""
    global num_inputs, num_hiddens1, num_hiddens2, num_outputs, num_epochs, lr, batch_size

    # Step 1：神经网络参数配置及权重初始化
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

    # Step 2：网络模型结构定义，几种不同的写法
    from collections import OrderedDict
    # method1
    net = nn.Sequential(
        OrderedDict([
          ('flatten', FlattenLayer()),
          ('linear1', nn.Linear(num_inputs, num_hiddens1)),
          ('relu1',   nn.ReLU()),
          ('dropout1', nn.Dropout(0.2)),
          ('linear2', nn.Linear(num_hiddens1,num_hiddens2)),
          ('relu2', nn.ReLU()),
          ('dropout2', nn.Dropout(0.5)),
          ('output', nn.Linear(num_hiddens2,num_outputs)),
          ('relu3', nn.ReLU()),
          ])
        )
    print(net, type(net))

    # Step 3：权重初始化
    # print("net.parameters:", list(net.parameters())) 
    for param in net.parameters():
        init.normal_(param, mean=0, std=0.01)

    # Step 4：神经网络超参数与数据集加载
    num_epochs, lr, batch_size = 1000, 0.3, 64
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # Step 5：损失函数
    loss = torch.nn.CrossEntropyLoss()

    # Step 6：定义优化算法
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    # optimizer = torch.optim.Adagrad(net.parameters(),lr=0.5)
    optimizer = torch.optim.Adadelta(net.parameters(),lr=0.5)
    # optimizer = torch.optim.Adam(net.parameters(),lr=0.5)
    
    # Step 7：模型训练
    train(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

    # Step 8：模型预测
    predict(net, test_iter)

if __name__ == '__main__':
    main()