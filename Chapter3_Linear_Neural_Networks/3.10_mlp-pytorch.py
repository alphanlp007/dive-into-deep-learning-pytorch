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
    # print("X shape:",X.shape)
    # print("y shape:",y.shape)
    y_hat = net(X).argmax(dim = 1)

    true_labels = get_fashion_mnist_labels(y.numpy())
    predict_labels = get_fashion_mnist_labels(y_hat.numpy())

    titles = [true + '\n' + pred for true, pred in zip(true_labels, predict_labels)]
    show_fashion_mnist(X[10:19], titles[10:19], imagepath = '../figures/3.10_mlp-pytorch.jpg')

def main():
    # config parameters 
    num_inputs = 784
    num_outputs = 10
    num_hiddens = 256
    batch_size = 64
    num_epochs = 30

    # Step 1：获取和读取数据
    train_iter, test_iter = load_data_fashion_mnist(batch_size = batch_size)
    # X, y = iter(train_iter).next()
    # print("X shape:",X.shape)
    # print("y shape:",y.shape)

    # Step 2：网络模型结构定义，几种不同的写法
    from collections import OrderedDict
    # method1
    net = nn.Sequential(
        # FlattenLayer(),
        # nn.Linear(num_inputs, num_outputs)
        OrderedDict([
          ('flatten', FlattenLayer()),
          ('linear1', nn.Linear(num_inputs, num_hiddens)),
          ('relu',nn.ReLU()),
          ('linear2',nn.Linear(num_hiddens,num_outputs))])
        )
    # # method2
    # net = nn.Sequential(
    #         FlattenLayer(),
    #         nn.Linear(num_inputs,num_hiddens),
    #         nn.ReLU(),
    #         nn.Linear(num_hiddens,num_outputs)
    # )
    print(net)

    # Step 3：权重初始化
    # print("net.parameters:",list(net.parameters())) # shape: [784,256],[256],[256,10],[10]
    for param in net.parameters():
        init.normal_(param, mean=0, std=0.01)
    # init.normal_(net.linear.weight, mean=0, std=0.01)
    # init.constant_(net.linear.bias, val=0) 

    # Step 4：定义损失函数
    loss = torch.nn.CrossEntropyLoss()

    # Step 5：定义优化算法
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    # optimizer = torch.optim.Adagrad(net.parameters(),lr=0.5)
    optimizer = torch.optim.Adadelta(net.parameters(),lr=0.5)
    # optimizer = torch.optim.Adam(net.parameters(),lr=0.5)

    # Step 6：神经网络训练
    train(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

    # Step 7：测试集预测
    predict(net, test_iter)

if __name__ == '__main__':
    main()