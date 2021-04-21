# -*- coding:utf-8 -*-
# 2021.04.17
# 多层感知机
import torch
import matplotlib.pyplot as plt

# 设置figure的大小, 默认画面大小为600 * 650
def set_size(figsize=(6, 6.5)):
    plt.figure(figsize = figsize)

# 绘制x和y的曲线图
def xyplot(x_vals, y_vals, name):
    """
    x_vals / y_vals: input tensor
    """
    set_size(figsize=(6,6.5)) # 设置figure图的大小，figsize为600 * 650
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy()) # detach()：Returns a new Tensor, detached from the current graph.
                                                               # The result will never require gradient。
    plt.xlabel('x')
    plt.ylabel(name + '(x)')

def main():
    print(torch.__version__)

    # Step 1：init x values，可以计算梯度值
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad = True)

    ### Relu激活函数 ###
    y = x.relu()
    xyplot(x, y, 'relu')
    
    # Relu的梯度
    y.sum().backward() # 标量对自变量x求梯度
    xyplot(x, x.grad, 'grad of relu')
    plt.savefig('../figures/3.8_mlp_relu.jpg')

    ### sigmoid激活函数 ###
    y = x.sigmoid()
    xyplot(x, y, 'sigmoid')

    x.grad.zero_() # 梯度清零，不然会造成梯度累加
    # Sigmoid的梯度
    y.sum().backward()
    xyplot(x, x.grad, 'grad of sigmoid')
    plt.savefig('../figures/3.8_mlp_sigmoid.jpg')

    ### tanh激活函数 ###
    y = x.tanh()
    xyplot(x, y, 'tanh')

    x.grad.zero_()
    # Tanh的梯度
    y.sum().backward()
    xyplot(x, x.grad, 'grad of tanh')
    plt.savefig('../figures/3.8_mlp_tanh.jpg')

if __name__ == "__main__":
    main()