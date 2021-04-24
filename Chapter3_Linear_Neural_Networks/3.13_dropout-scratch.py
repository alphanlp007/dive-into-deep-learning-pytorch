# -*- coding:utf-8 -*-
# 2021.04.24
import torch
import numpy as np

# 导入当前目录的父目录的package
import os,sys
sys.path.append("..")
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils.utils import show_fashion_mnist, get_fashion_mnist_labels, load_data_fashion_mnist

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

def net(X, is_training = True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:  # 只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层
    return torch.matmul(H2, W3) + b3


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
    show_fashion_mnist(X[10:19], titles[10:19], imagepath = '../figures/3.13_dropout.jpg')

def main():
    """begin 神经网络参数配置"""
    global num_inputs, num_hiddens1, num_hiddens2, num_outputs, \
        drop_prob1, drop_prob2, num_epochs, lr, batch_size

    global net, W1, b1, W2, b2, W3, b3

    # 神经网络参数配置及权重初始化
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

    W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
    b1 = torch.zeros(num_hiddens1, requires_grad=True)
    W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
    b2 = torch.zeros(num_hiddens2, requires_grad=True)
    W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
    b3 = torch.zeros(num_outputs, requires_grad=True)

    # 待优化参数
    params = [W1, b1, W2, b2, W3, b3]

    # 给不同隐藏层输出，设置不同的dropout参数值
    drop_prob1, drop_prob2 = 0.2, 0.5

    # 神经网络超参数
    num_epochs, lr, batch_size = 50, 100.0, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # 损失函数
    loss = torch.nn.CrossEntropyLoss()

    # 网络模型
    net = net
    """end 神经网络参数配置"""
    
    # 模型训练
    train(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

    # 模型预测
    predict(net, test_iter)

if __name__ == '__main__':
    main()