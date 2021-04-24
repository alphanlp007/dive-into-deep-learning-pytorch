# -*- coding:utf-8 -*-
# 2021.04.18

import torch
import numpy as np
import matplotlib.pyplot as plt

# 绘图函数
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(10, 8), imagepath='../figures/3.11_underfit-overfit.jpg'):
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

# 生成数据集
def data_iter():
    # 1、定义训练数据和测试数据的样本数量
    # 2、定义权重参数
    # 3、定义偏移参数
    n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5

    # 特征生成
    features = torch.randn(n_train + n_test, 1) # shape:[n_train + n_test, 1]
    poly_features = torch.cat((features, torch.pow(features,2), torch.pow(features,3)), dim=1) # [x x^2 x^3]
    # print(features.shape)
    # print(poly_features.shape)

    # 标签生成
    # y = 1.2*x - 3.4*x^2 + 5.6*x^3 + 5 + β
    labels = true_w[0] * poly_features[:,0] + true_w[1] * poly_features[:,1] + true_w[2] * poly_features[:,2] + true_b
    # print(labels.size())

    # 标签添加噪声
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

    return features, poly_features, labels

def fit_and_plot(train_features, test_features, train_labels, test_labels, num_epochs, loss):
    # a linear transformation layer
    net = torch.nn.Linear(train_features.shape[-1], 1)

    batch_size = min(10, train_features.shape[0])

    # 数据集特征features与标签labels组合
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
    train_loss, test_loss = [],[]

    for epoch in range(num_epochs):
        for X,y in train_iter:
            loss_ = loss(net(X), y.view(-1,1)) # 计算batch损失
            optimizer.zero_grad()              # 梯度清零
            loss_.backward()                   # 反向传播求梯度
            optimizer.step()                   # 权重参数更新
        train_labels = train_labels.view(-1,1)
        test_labels  = test_labels.view(-1,1)

        train_loss.append(loss(net(train_features), train_labels).item())
        test_loss.append(loss(net(test_features), test_labels).item())

    print("final epoch: train_loss:{:.4f}".format(train_loss[-1]),' test_loss:{:.4f}'.format(test_loss[-1]))
    semilogy(range(1, num_epochs + 1), train_loss, 'epochs', 'loss', range(1, num_epochs + 1), test_loss, ['train', 'test'])
    print('weight:', net.weight.data, '\nbias:', net.bias.data)

def main():
    print(torch.__version__)
    features, poly_features, labels = data_iter()
    # print(features[:10],poly_features[:10],labels[:10])

    # example
    # semilogy([1,2], [2.3,3.3], 'epochs', 'loss', range(1,3), [2.3,3.3], ['train', 'test'])
    # semilogy(range(1,3), [2.3,3.3], 'epochs', 'loss', range(1,3), [2.3,3.3], ['train', 'test'])

    num_epochs = 100
    loss = torch.nn.MSELoss()

    # 多项式拟合
    fit_and_plot(poly_features[:100, :], poly_features[100:, :], labels[:100], labels[100:],num_epochs,loss)
    
    # # 线性函数拟合（欠拟合）
    # fit_and_plot(features[:100, :], features[100:, :], labels[:100], labels[100:],num_epochs,loss)

    # # 训练样本不足（过拟合）
    # fit_and_plot(poly_features[0:2, :], poly_features[100:, :], labels[0:2], labels[100:],num_epochs,loss)

if __name__ == "__main__":
    main()