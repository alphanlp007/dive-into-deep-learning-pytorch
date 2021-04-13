# -*- coding:utf-8 -*-
# 2021.04.12
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 导入当前目录的父目录的package
import os,sys
sys.path.append("..")
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils.utils import show_fashion_mnist, get_fashion_mnist_labels, load_data_fashion_mnist

def main():
    # Step 1：config parameters
    batch_size = 256
    num_inputs = 784
    num_outputs = 10

    num_epochs = 30
    lr = 0.1

    # Step 2：加载训练数据和测试数据迭代对象
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    # 数据集测试
    from matplotlib import pyplot as plt
    for X, y in train_iter:
        print('Image:',X.shape,'label:',y.shape)
        img = X[0].view(28,28).numpy()
        label = get_fashion_mnist_labels([y[0]])

        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
        plt.savefig('../figures/3.6_train_iter_test.jpg')
        
        break

if __name__ == '__main__':
    main()