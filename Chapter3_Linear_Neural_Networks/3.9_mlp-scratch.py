# -*- coding:utf-8 -*-
# 2021.04.17
import torch
from torch.nn import init
from torch import nn
import numpy as np

# 导入当前目录的父目录的package
import os,sys
sys.path.append("..")
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils.utils import show_fashion_mnist, get_fashion_mnist_labels, load_data_fashion_mnist

# 神经网络权重参数初始化
def init_weight(num_inputs, num_hiddens, num_outputs):
    # 隐藏层
    W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs,num_hiddens)), dtype=torch.float, requires_grad=True)
    b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)

    # 输出层
    W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float, requires_grad=True)
    b2 = torch.zeros(num_outputs,dtype=torch.float, requires_grad=True)

    # 网络权重参数和偏置量
    params = [W1, b1, W2, b2]
    
    return params

# 定义激活函数
def relu(X):
    """
    the tensor X size is [batch_size, num_hiddens]
    Each element of the tensor input is compared with the corresponding element 
    of the tensor other and an element-wise maximum is taken.
    The shapes of input and other don’t need to match, but they must be broadcastable.
    out_i = max(tensor_i,other_i)
    """
    return torch.max(X, other=torch.tensor(0.0))

# 神经网络模型定义
def net(X):
    # 输入层
    X = X.view(-1, num_inputs) # [256, 1, 28, 28] ==> [256, 784]

    # 隐藏层
    H = relu(torch.matmul(X, W1) + b1)

    # 输出层
    y = torch.matmul(H, W2) + b2

    # 结果返回
    return y

# 定义随机梯度下降优化算法
def sgd(params, lr, batch_size):
    for param in params:
        # 1）参数使用param.data，获取纯数据
        # 2) 梯度为何要处理batch_size，梯度是mini batch上梯度的累加，所以要除以batch_size的大小
        param.data -= lr * param.grad / batch_size

def accuracy(y_hat, y):
    bool_res = (y_hat.argmax(dim = 1) == y) # tensor([False,  True])
    float_res = bool_res.float()            # tensor([0., 1.])
    scalar_value = float_res.mean().item()  # 0.5
    return scalar_value

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim = 1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n # 除以总样本量是否正确？ ==>解释：预测正确的总数量，除以总的测试样本数

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
            loss_ = loss(y_hat, y).sum() # 当前batch的交叉熵总和，即损失值

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

def predict(net,test_iter):
    X, y = iter(test_iter).next()
    print("X shape:",X.shape)
    print("y shape:",y.shape)
    y_hat = net(X).argmax(dim = 1)

    true_labels = get_fashion_mnist_labels(y.numpy())
    predict_labels = get_fashion_mnist_labels(y_hat.numpy())

    titles = [true + '\n' + pred for true, pred in zip(true_labels, predict_labels)]
    show_fashion_mnist(X[10:19], titles[10:19], imagepath = '../figures/3.9_mlp-scratch.jpg')

def main():
    # 网络参数全局配置
    global W1, b1, W2, b2, num_inputs

    # config parameters 
    num_inputs = 784
    num_outputs = 10
    num_hiddens = 256
    batch_size = 64
    num_epochs = 10
    lr = 0.5 

    # 获取和读取数据
    train_iter, test_iter = load_data_fashion_mnist(batch_size = batch_size)

    # 初始化神经网络模型参数
    params = init_weight(num_inputs, num_hiddens, num_outputs)
    [W1, b1, W2, b2] = params

    # 定义损失函数和优化器
    loss = torch.nn.CrossEntropyLoss()

    # 神经网络训练
    train(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

    # 测试集预测
    predict(net, test_iter)

if __name__ == '__main__':
    main()