# -*- coding:utf-8 -*-
import torch
import pandas as pd
import torch.nn as nn

# 导入当前目录的父目录的package
import os,sys
sys.path.append("..")
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils.utils import semilogy

def load_data(path):
    """数据集加载"""
    # check path is available
    if not os.path.exists(path):
        raise FileExistsError('文件路径不存在。')

    train_path = os.path.join(path, 'train.csv')
    test_path  = os.path.join(path, 'test.csv')

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, test_data

def preprocess_data(train_data,test_data):
    """
    description: 对数据集进行归一化和指示特征编码
    Input:
        输入csv文件提取的训练数据和测试数据
    Output：
        输出经归一化和指示特征encoding编码后的特征数据
    """
    # print(train_data.shape)
    # print(test_data.shape)
    n_train = train_data.shape[0]  # (1460, 81)

    # Step 1：样本数据拼接
    # 训练数据集特征与测试数据集特征拼接
    # 训练数据集：去掉ID编号和label标签
    # 测试数据集：去掉ID编号
    all_features = pd.concat([train_data.iloc[:,1:-1],test_data.iloc[:,1:]])
    # print(all_features.shape)    # (2919, 79)

    # Step 2：对样本数据中的数值类特征做归一化处理
    # 特征类型有数值型特征和指示性特征
    # 该步骤主要提取出数值型特征，进行特征归一化处理
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    
    # 计算均值和方差进行特征归一化
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    
    # 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # Step 3：指示特征(one-hot encoding 和 dummy encoding)
    # dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
    # 函数pandas.get_dummies()的用法 https://blog.csdn.net/maymay_/article/details/80198468
    all_features = pd.get_dummies(all_features, dummy_na=True) # dummy_na:用于标识样本数据中是否包含NAN项

    train_data_features = all_features.iloc[:n_train]
    test_data_features  = all_features.iloc[n_train:]
    # print(train_data_features.shape)
    # print(test_data_features.shape)

    # 数值转换为Tensor类型
    train_data_features = torch.tensor(train_data_features.values, dtype=torch.float)
    test_data_features  = torch.tensor(test_data_features.values, dtype=torch.float)

    return train_data_features, test_data_features

# 网络搭建
def get_net(num_features):
    """网络搭建和权重参数初始化，线性模型"""
    net = torch.nn.Linear(num_features, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0,std=0.01)
    return net

loss = nn.MSELoss() # MSELoss：均方损失函数，预测值和真实值之间差的平方和的平均数
def log_rmse(net, features, labels): # 
    with torch.no_grad():  # 不进行计算图的构建
        # 将小于1的值设置成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        
        # 均方根损失函数，预测结果取对数后，再计算损失
        loss_ = loss(clipped_preds.log(), labels.log())
        rmse = torch.sqrt(loss_)

    return rmse.item()

def train(net, train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    """
    功能：
        神经网络模型的训练
    """
    train_loss, test_loss = [],[]

    # 训练数据，特征与标签进行组合
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    # 优化器定义
    optimizer = torch.optim.Adam(params=net.parameters(),lr=learning_rate,weight_decay=weight_decay)

    net = net.float()

    # 训练
    for epoch in range(num_epochs):
        for X, y in train_iter:
            loss_ = loss(net(X.float()), y.float())
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss_.backward()
            # 权重参数更新
            optimizer.step()

        # 训练误差
        train_loss.append(log_rmse(net,train_features,train_labels))

        if test_labels is not None:
            test_loss.append(log_rmse(net,test_features,test_labels))

        # print("epoch = %d, train_loss：%.4f, test_loss：%.4f" % (epoch+1, log_rmse(net,train_features,train_labels), log_rmse(net,test_features,test_labels)))

    return train_loss, test_loss

def k_fold_train(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    """
    功能：
        K折交叉训练
    """
    train_loss_num, valid_loss_num = 0, 0
    net = None
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)         # X_train, y_train, X_valid, y_valid
        net = get_net(X_train.shape[1])
        train_loss, valid_loss = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)

        train_loss_num += train_loss[-1]
        valid_loss_num += valid_loss[-1]

        if i == 0:
            semilogy(range(1,num_epochs+1), train_loss, 'epochs', 'rmse', range(1,num_epochs+1), valid_loss, ['train','valid'],\
                     imagepath='../figures/3.16_kaggle-house-price.jpg')

        print('fold %d, train rmse %f, valid rmse %f' % (i, train_loss[-1], valid_loss[-1]))

    return train_loss_num / k, valid_loss_num / k, net

def get_k_fold_data(k, i, X, y):
    """
    功能：
        k折交叉验证，k表示将数据切分的份数,i表示选取的第i份数据作为验证集
    返回：
        返回第i折交叉验证时所需要的训练和验证数据
    """
    assert k > 1
    fold_size = X.shape[0] // k             # 9//2 = 4
    X_train, y_train = None, None
    X_valid, y_valid = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j+1)*fold_size)
        X_part, y_part = X[idx,:], y[idx]   # 数据索引
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    """
    功能：
        预测并生成符合Kaggle格式的csv文件
    """
    net = get_net(train_features.shape[1])
    # 网络训练
    train_loss, _ = train(net, train_features, train_labels, None,None,num_epochs,lr,weight_decay,batch_size)
    semilogy(range(1,num_epochs+1),train_loss,'epochs','rmse')
    print("train rmse %f" % train_loss[-1])
    
    # 使用训练好的网络，进行预测
    preds = net(test_features).detach().numpy()

    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./figures/submission.csv', index=False)

def predict(net,test_features, test_data):
    """
    功能：
        使用训练好的模型，在测试集上进行预测
    """
    # 使用训练好的网络，进行预测
    preds = net(test_features).detach().numpy()

    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('../figures/submission.csv', index=False)


def main():
    # Step 1：数据集加载
    train_data, test_data = load_data('./kaggle_house/')

    # print(test_data.iloc[0:4,[0,1,2,3,-2,-1]])
    # print(test_data.columns.values) # 打印列的表头

    # Step 2：数据集特征预处理
    train_data_features, test_data_features = preprocess_data(train_data, test_data)        # 训练数据集特征和测试数据集特征
    train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1,1)  # 训练集标签

    print("data from csv: ",train_data.shape, test_data.shape)
    print("process data: ",train_data_features.shape, test_data_features.shape)
    print("train labels: ",train_labels.shape)

    # Step3：模型超参数
    k, num_epochs, lr, weight_decay, batch_size = 10, 100, 5, 0.1, 64
    
    # method1 模型训练和验证，k折交叉验证训练
    train_loss, valid_loss, net = k_fold_train(k, train_data_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_loss, valid_loss))

    # method2 模型训练和测试（包含模型的重新训练）
    # train_and_pred(train_data_features,test_data_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size)

    # 直接使用训练好的模型，在测试集上进行预测
    predict(net, test_data_features, test_data)

if __name__ == '__main__':
    main()