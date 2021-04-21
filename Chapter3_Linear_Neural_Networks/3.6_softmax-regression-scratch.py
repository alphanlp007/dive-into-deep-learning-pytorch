# -*- coding:utf-8 -*-
# 2021.04.12
import torch
import numpy as np

# 导入当前目录的父目录的package
import os,sys
sys.path.append("..")
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils.utils import show_fashion_mnist, get_fashion_mnist_labels, load_data_fashion_mnist

def net(X):
    """
    单层神经softmax神经网络，有多个分类输出
    X：the input tensor, size is [batch_size, num_inputs]
    W: the weight tensor, size is [num_inputs, class]
    b: the bias tensor, value is constant zero
    """
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
    # # unit test
    # y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    # y = torch.tensor([0, 2])
    # print(y_hat.view(-1,3))
    # print(y.view(-1,1))

    # torch.gather(input, dim, index, out=None) 
    # 沿给定轴dim，提取指定index位置的值,
    # 例如：y_hat为softmax输出的概率分布[0.1 0.02 0.3 ... 0.5]，由于样本只属于一个类别
    # 则单个样本的损失函数loss = -log(y_hat_yi)
    loss = torch.gather(y_hat, 1, y.view(-1, 1))
    return -torch.log(loss) # 计算batch中每个样本的交叉熵损失

def sgd(params, lr, batch_size):
    # 定义随机梯度下降优化算法
    for param in params:
        # 1）参数使用param.data，获取纯数据
        # 2) 梯度为何要处理batch_size，梯度是mini batch上梯度的累加，所以要除以batch_size的大小
        param.data -= lr * param.grad / batch_size

def accuracy(y_hat, y):
    # 精度率统计
    bool_res = (y_hat.argmax(dim = 1) == y) # tensor([False,  True])
    float_res = bool_res.float()            # tensor([0., 1.])
    scalar_value = float_res.mean().item()  # 0.5
    return scalar_value

def evaluate_accuracy(data_iter, net):
    # 在数据集上统计精确率
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim = 1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n # 除以总样本量是否正确？ ==> 解释：预测正确的总数量，除以总的测试样本数，混淆矩阵中的TP

def init_weight(num_inputs, num_outputs):
    # W权重初始化为均值为0，方差为0.01的张量矩阵
    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs,num_outputs)), dtype=torch.float)
    b = torch.zeros(num_outputs, dtype=torch.float)

    W.requires_grad_(requires_grad = True)
    b.requires_grad_(requires_grad = True)

    return W, b

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

def predict(net, test_iter):
    X, y = iter(test_iter).next()
    y_hat = net(X).argmax(dim = 1)

    true_labels = get_fashion_mnist_labels(y.numpy())
    predict_labels = get_fashion_mnist_labels(y_hat.numpy())

    titles = [true + '\n' + pred for true, pred in zip(true_labels, predict_labels)]
    show_fashion_mnist(X[10:19], titles[10:19], imagepath='../figures/3.6_softmax-regression-scratch.jpg')

def main():
    # 全局变量配置, Python中的全局变量只能先定义后赋值
    global num_inputs, W, b

    # Step 1：config parameters
    batch_size = 256
    num_inputs = 784
    num_outputs = 10

    num_epochs = 30
    lr = 0.1

    # Step 2：加载训练数据和测试数据迭代对象
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    # # 数据集测试
    # from matplotlib import pyplot as plt
    # for X, y in train_iter:
    #     print('Image:',X.shape,'label:',y.shape)
    #     img = X[0].view(28,28).numpy()
    #     label = get_fashion_mnist_labels([y[0]])

    #     plt.imshow(img)
    #     plt.title(label)
    #     plt.axis('off')
    #     plt.savefig('../figures/3.6_train_iter_test.jpg')

    #     break

    # Step 3：网络权重和偏移量初始化
    W, b = init_weight(num_inputs, num_outputs)

    # Step 4：调用训练函数
    train(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, params=[W,b], lr = lr)

    # Step 5：测试集评价模型
    predict(net, test_iter)

if __name__ == '__main__':
    main()