# -*- coding:utf-8 -*-
# 2021.04.17
import torch
from torch.nn import init
from torch import nn

# 导入当前目录的父目录的package
import os,sys
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

def sgd(params, lr, batch_size):
    # 定义随机梯度下降优化算法
    for param in params:
        # 1）参数使用param.data，获取纯数据
        # 2) 梯度为何要处理batch_size，梯度是mini batch上梯度的累加，所以要除以batch_size的大小
        param.data -= lr * param.grad / batch_size

def evaluate_accuracy(data_iter, net):
    # 在数据集上统计精确率
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim = 1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n # 除以总样本量是否正确？ ==> 解释：预测正确的总数量，除以总的测试样本数，混淆矩阵中的TP

def train(net, train_iter, test_iter, loss, num_epochs, batch_size,
          params = None, lr = None, optimizer = None):
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            loss_ = loss(y_hat, y).sum() # 当前batch的交叉熵总和

            ### 梯度清零，否则会出现梯度累加，再进行反向传播计算网络参数的梯度 ###
            if optimizer is not None: 
                # 使用Pytorch优化器
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None: 
                # 使用自己编写的优化器
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
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item() # 累加每个batch预测正确的样本数量
            n += y.shape[0]

        # 单个epoch训练结束，评估模型在测试集上的精度
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss：%.4f, train_acc %.4f, test_acc %.4f"\
            % (epoch+1, train_loss_sum/n, train_acc_sum/n, test_acc))

def predict(net, test_iter):
    X, y = iter(test_iter).next()
    y_hat = net(X).argmax(dim = 1)

    true_labels = get_fashion_mnist_labels(y.numpy())
    predict_labels = get_fashion_mnist_labels(y_hat.numpy())

    titles = [true + '\n' + pred for true, pred in zip(true_labels, predict_labels)]
    show_fashion_mnist(X[10:19], titles[10:19], imagepath='../figures/3.7_softmax-regression-pytorch.jpg')

def main():
    # 训练数据集和测试数据集加载
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    X, y = iter(train_iter).next()
    print("X shape:",X.shape)
    print("y shape:",y.shape)

    from collections import OrderedDict
    # 网络定义
    num_inputs = 784
    num_outputs = 10
    net = nn.Sequential(
        # FlattenLayer(),
        # nn.Linear(num_inputs, num_outputs)
        OrderedDict([
          ('flatten', FlattenLayer()),
          ('linear', nn.Linear(num_inputs, num_outputs))])
        )

    # print("FlattenLayer:",net[0],'\nLinear:',net[1])

    # 权重初始化
    init.normal_(net.linear.weight, mean=0, std=0.01)
    init.constant_(net.linear.bias, val=0) 

    # 定义损失函数
    loss = nn.CrossEntropyLoss()

    # 定义优化算法
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 训练模型
    num_epochs = 30
    train(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

    # 模型在测试集上的预测
    predict(net, test_iter)

if __name__ == '__main__':
    main()