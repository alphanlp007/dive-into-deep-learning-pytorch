# -*- coding:utf-8 -*-
# 2021.04.11
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 导入当前目录的父目录的package
import os,sys
sys.path.append("..")
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils.utils import show_fashion_mnist, get_fashion_mnist_labels

def data_iter(batch_size, data, shuffle = True):
    """
    batch_size：批量大小
    data：元组数据(features, labels)
    """
    import sys
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 1

    data_iters = torch.utils.data.DataLoader(dataset = data, 
                                             batch_size = batch_size, 
                                             shuffle = shuffle, 
                                             num_workers = num_workers
                                            )
    
    return data_iters

def main():
    # Step 1：获取数据集
    mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/', 
                                                    train=True, 
                                                    download=True, 
                                                    transform=transforms.ToTensor()
                                                    )
    mnist_test  = torchvision.datasets.FashionMNIST(root='./Datasets/', 
                                                    train=False, 
                                                    download=True, 
                                                    transform=transforms.ToTensor()
                                                    )

    ### begin Unit Test ###
    # output FashionMNIST
    print(type(mnist_train))
    print(len(mnist_train), len(mnist_test))

    feature, label = mnist_train[0]
    print("feature.shape:", feature.shape, '\n'+"feature.dtype:", feature.dtype) # Channel * Height * Width -> torch.Size([1, 28, 28]) 
    print("label:",label)                                                        # 9 -> torch.float32
    ### end Unit Test ###

    X, y = [], []
    for i in range(10):
        X.append(mnist_train[i][0])  # features
        y.append(mnist_train[i][1])  # labels
    show_fashion_mnist(X, get_fashion_mnist_labels(y),'../figures/test.jpg') 

    train_iter = data_iter(batch_size = 256, data = mnist_train, shuffle = True)
    test_iter  = data_iter(batch_size = 256, data = mnist_test,  shuffle = False)

    import time
    start = time.time()
    for X, y in train_iter:
        continue
    print("%.2f sec" % (time.time() - start))

if __name__ == "__main__":
    main()