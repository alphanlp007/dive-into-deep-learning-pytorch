# -*- coding:utf-8 -*-
# 2021.04.10

import torch
import numpy as np

def generateData(nums_samples, nums_features):
    """生成模拟数据集 y = X*w + b"""
    true_w = [2, -3.4]
    true_b = 4.2

    # 标准正态分布生成特征，均值为0，方差为1
    features = torch.randn(nums_samples, nums_features, dtype=torch.float32)
    labels   = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b

    # 添加均值为0，方差为0.01的噪声项
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                           dtype=torch.float32)
    
    return features, labels

def main():
    # check torch version
    print("pytorch version:", torch.__version__)

    features, labels = generateData(1000, 2)

if __name__ == '__main__':
    main()