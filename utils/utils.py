# -*- coding:utf-8 -*-
# 2021.04.11
import torch
import random
from IPython import display
from matplotlib import pyplot as plt

############ 3.2_linear-regression-scratch ############
def use_svg_display():
    # 使用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5),):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def data_iter(batch_size, features, labels):
    """
    batch_size:批量大小
    features:特征
    labels:样本标签
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        start = i
        end   = min(i+batch_size, num_examples)
        # 样本索引
        j     = torch.LongTensor(indices[start:end])

        # 按行索引
        yield features.index_select(0, j), labels.index_select(0, j)

def linear_regression(X, w, b):
    """线性回归模型"""
    # result = torch.mm(X, w) + b
    temp = torch.matmul(X, w) + b
    return temp

def squared_loss(y_hat, y):
    """平方损失函数"""
    loss = (y_hat - y.view(y_hat.size())) ** 2/2
    return loss

def sgd(params, lr, batch_size):
    """随机梯度下降优化算法"""
    for param in params:
        # 梯度需要除以batch_size的大小，当前损失的平均梯度
        # 注意这里更改param时用的param.data
        param.data -= lr * param.grad / batch_size 


############ 3.5_fashion-mnist.py ############
def show_fashion_mnist(images, labels, imagepath='../figures/result.jpg'):
    from IPython import display
    """Use svg format to display plot"""
    display.set_matplotlib_formats('svg')
    _, figs = plt.subplots(1, len(images), figsize = (12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view(28, 28).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)       
    # plt.show() # linux系统不支持图像显示
    plt.savefig(imagepath)

def get_fashion_mnist_labels(labels):
    """将数字标签转换为文本标签"""
    text_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return [text_labels[int(i)] for i in labels]
