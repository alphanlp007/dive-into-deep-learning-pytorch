# -*- coding:utf-8 -*-
# 2021.04.18

import torch
import numpy as np
import matplotlib.pyplot as plt

# 绘图函数
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(10, 8), imagepath='../figures/3.12_weight-decay.jpg'):
    """
    x_vals: 原始数据x
    y_vals: 原始数据y
    x_labels: x轴标签
    y_labels: y轴标签
    x2_vals: 对比数据x
    y2_vals: 对比数据y
    legend: 数据曲线标签
    matplotlib：plt.semilogx(*args, **kwargs) 和 plt.semilogy(*args, **kwargs)
    描述：用于绘制折线图，两个函数的 x 轴、y 轴分别是指数型的
    """
    plt.figure(figsize=figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    
    try:
        plt.savefig(imagepath)
    except FileNotFoundError:
        raise FileNotFoundError("没有找到要保存的路径。")

# 参数初始化
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义L2范数惩罚项
def l2_penalty(w):
    """
    权重衰减，对所有的权重w，计算平方和，即1/2 * ||w||^2
    """
    return (w**2).sum() / 2


# 定义线性回归模型
def line_regression(X, w, b):
    """
    X: [batch_size, num_inputs]
    W: [num_inputs, 1]
    b: [1], 广播机制
    """
    return torch.mm(X, w) + b


# 定义损失函数，平方损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        # 1）参数使用param.data，获取纯数据
        # 2) 梯度为何要处理batch_size，梯度是mini batch上梯度的累加，所以要除以batch_size的大小
        param.data -= lr * param.grad / batch_size

# 从零开始实现线性拟合
def fit_and_plot(train_iter, train_features, train_labels, test_features, test_labels, lambd = 0):
    w, b = init_params()
    train_loss, test_loss = [], []
    for epoch in range(num_epochs):
        train_loss_sum, n = 0.0, 0
        for X, y in train_iter:
            y_hat = line_regression(X, w, b) # 线性回归预测
            loss_ = squared_loss(y_hat, y) + lambd * l2_penalty(w) # 未对损失函数计算平均值
            
            loss_mean = loss_.mean()    # 损失函数计算平均值
            loss_ = loss_.sum()

            if w.grad is not None: # 梯度清零
                w.grad.data.zero_()
                b.grad.data.zero_()

            loss_mean.backward() # 梯度计算
            sgd([w, b], lr, batch_size) # 权重更新

            train_loss_sum += loss_.item()
            n += y.shape[0]

        print("epoch %d, loss：%.4f" % (epoch+1, train_loss_sum/n))

        train_loss.append(squared_loss(line_regression(train_features, w, b), train_labels).mean().item())
        test_loss.append(squared_loss(line_regression(test_features, w, b), test_labels).mean().item())

    semilogy(range(1, num_epochs + 1), train_loss, 'epochs', 'loss', \
             range(1, num_epochs + 1), test_loss, ['train', 'test'], \
             imagepath='../figures/3.12_weight-decay-scrath.jpg')
    
    print('L2 norm of w:', w.norm().item()) # 权重的L2范数

# pytorch实现线性拟合
def fit_and_plot_pytorch(train_iter, train_features, train_labels, test_features, test_labels, lambd = 0):
    net = torch.nn.Linear(num_inputs,1)
    torch.nn.init.normal_(net.weight, mean=0, std=1)
    torch.nn.init.normal_(net.bias, mean=0, std=1)

    optimizer_w = torch.optim.SGD(params=[net.weight], lr = lr, weight_decay = lambd)
    optimizer_b = torch.optim.SGD(params=[net.bias], lr = lr)

    train_loss, test_loss = [], []
    for epoch in range(num_epochs):
        train_loss_sum, n = 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)  # 线性回归预测
            loss_ = squared_loss(y_hat, y).mean() # 当前batch的平均损失

            # 梯度清零
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            
            # 反向传播
            loss_.backward()

            # 对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()

            train_loss_sum += loss_.item()*y.shape[0]
            n += y.shape[0]

        print("epoch %d, loss：%.4f" % (epoch+1, train_loss_sum/n))

        train_loss.append(squared_loss(net(train_features), train_labels).mean().item())
        test_loss.append(squared_loss(net(test_features), test_labels).mean().item())

    semilogy(range(1, num_epochs + 1), train_loss, 'epochs', 'loss', \
             range(1, num_epochs + 1), test_loss, ['train', 'test'], \
             imagepath='../figures/3.12_weight-decay-pytorch.jpg')
    
    print('L2 norm of w:',net.weight.data.norm().item()) # 权重的L2范数


def main():
    # 全局变量定义
    global batch_size, num_epochs, lr, num_inputs
    batch_size = 64
    num_epochs = 100
    lr = 0.003

    # 高维线性回归实验
    n_train, n_test, num_inputs = 500, 100, 200
    true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
    
    features = torch.randn(n_train + n_test, num_inputs) # 200维特征
    print(features.shape)

    # y = 0.05 + Σ 0.01*x + β
    labels  = torch.matmul(features, true_w) + true_b  # [600, 200]*[200, 1] + [600, 1] ===> [600, 1]
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

    train_features, test_features = features[:n_train, :], features[n_train:, :]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    # CPU线程数量判断
    import sys
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 1

    # 训练数据集的输入特征与训练数据集的标签进行组合
    dataset     = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter  = torch.utils.data.DataLoader(dataset = dataset, 
                                              batch_size = batch_size,
                                              shuffle=True,
                                              num_workers = num_workers
                                             )
    
    # 调用从零开始实现接口
    print("调用从零开始实现接口")
    # fit_and_plot(train_iter, train_features, train_labels, test_features, test_labels, lambd = 0)
    fit_and_plot(train_iter, train_features, train_labels, test_features, test_labels, lambd = 6)

    # 调用pytorch实现接口
    print("调用pytorch实现接口")
    # fit_and_plot_pytorch(train_iter, train_features, train_labels, test_features, test_labels, lambd = 0)
    # fit_and_plot_pytorch(train_iter, train_features, train_labels, test_features, test_labels, lambd = 6)

if __name__ == '__main__':
    main()