# -*- coding: utf-8 -*-
"""
@Time ： 2022/03/01 11:34
@Author ：KI 
@File ：CNN.py
@Motto：Hungry And Humble

"""
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from data_process import load_data
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed): #生成随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20) #设置随机数种子


class cnn(nn.Module):
    def __init__(self): # 初始化
        super(cnn, self).__init__()
        self.relu = nn.ReLU()  #设置激活函数relu
        self.sigmoid = nn.Sigmoid() #设置激活函数Sigmoid
        self.conv1 = nn.Sequential( # 添加模块
            nn.Conv2d( # 二维输入的神经网络卷积层
                in_channels=3, # 输入通道数为3，因为图片是彩色的，rgb类型
                out_channels=16, # 输出通道数为16
                kernel_size=3, #卷积核为3*3
                stride=2, # 卷积步长为2
            ),
            nn.BatchNorm2d(16), # 对数据进行归一化处理
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 最大池化操作，其中kernel_size为窗口大小
        )
        #
        self.conv2 = nn.Sequential(  # 第二个卷积层
            nn.Conv2d(
                in_channels=16, # 输入通道为16，因为上一个卷积层的输出通道为16
                out_channels=32, # 输出通道为32
                kernel_size=3, # 卷积核为3*3
                stride=2, # 步长为2
            ),
            nn.BatchNorm2d(32), # 数据进行归一化处理
            nn.ReLU(), # 使用relu为激活函数
            nn.MaxPool2d(kernel_size=2), #最大池化操作，其中kernel_size为窗口大小
        )
        #
        self.conv3 = nn.Sequential(  # 第三个卷积层
            nn.Conv2d(
                in_channels=32, #输入通道为32，因为上一个卷积层的输出通道为32
                out_channels=64, # 输出通道为64
                kernel_size=3, #卷积核为3*3
                stride=2, # 步长为2
            ),
            nn.BatchNorm2d(64), # 数据进行归一化处理
            nn.ReLU(),  # 使用relu为激活函数
            nn.MaxPool2d(kernel_size=2), #最大池化操作，其中kernel_size为窗口大小
        )

        self.fc1 = nn.Linear(3 * 3 * 64, 64) # 全连接层 in_feature输入大小为3*3*64,out_feature为64，也是神经元的个数
        self.fc2 = nn.Linear(64, 32) # 全连接层 in_feature输入为64,out_feature为32，也是神经元的个数
        self.out = nn.Linear(32, 16) # 输出层，输出类别为16类

    def forward(self, x): # 前向传播
        x = self.conv1(x) # 经过第一个卷积层
        x = self.conv2(x) # 经过第二个卷积层
        x = self.conv3(x) # 经过第三个卷积层
        # print(x.size())
        x = x.view(x.shape[0], -1) # 把二维特征变为一维特征，这样全连接层才可以处理
        x = self.relu(self.fc1(x)) # 使用relu作为激活函数
        x = self.relu(self.fc2(x)) # 使用relu作为激活函数
        # x = self.sigmoid(self.out(x))
        x = F.log_softmax(x, dim=1) # 计算损失的，将结果取对数？
        return x


def get_val_loss(model, Val): # 计算评估阶段的损失
    model.eval() # 将模型设置为评估模式
    criterion = nn.CrossEntropyLoss().to(device) # 计算交叉熵损失
    val_loss = [] # 存放损失的list
    for (data, target) in Val: # data是图片，target是图片的类别
        data, target = Variable(data).to(device), Variable(target.long()).to(device) # 创建变量
        output = model(data)  # 将data放到模型中进行计算，得到结果
        loss = criterion(output, target) # 利用交叉熵就算损失
        val_loss.append(loss.cpu().item()) # 将损失添加到val_loss中

    return np.mean(val_loss)


def train():
    Dtr, Val, Dte = load_data() # 将之前处理好的数据下载下来
    print('train...') # 开始训练
    epoch_num = 100 # 设置回合数
    best_model = None
    min_epochs = 5 # 最小回合数
    min_val_loss = 5 # 最小评估孙叔
    model = cnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0008) # 选取优化器为Adam
    criterion = nn.CrossEntropyLoss().to(device) #计算交叉熵损失
    # criterion = nn.BCELoss().to(device)
    for epoch in tqdm(range(epoch_num), ascii=True):
        train_loss = [] # 训练损失
        for batch_idx, (data, target) in enumerate(Dtr, 0): # enumerate的功能是计数，batch_idx就是在记录当前取了多少数据，索引从0开始
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            # target = target.view(target.shape[0], -1)
            # print(target)
            optimizer.zero_grad() # 梯度下降
            output = model(data)  # 将data放到模型中进行计算，得到结果
            # print(output)
            loss = criterion(output, target) # 计算损失
            loss.backward() # 反向传播计算得到每个参数的梯度值
            optimizer.step()  # 梯度下降执行一步参数更新
            train_loss.append(loss.cpu().item()) # 添加损失结果
        # validation
        val_loss = get_val_loss(model, Val) # 得到评估的损失
        model.train() # 调整模型为训练模式（因为在上一步中设成评估模式了）
        if epoch + 1 > min_epochs and val_loss < min_val_loss: # 如果评估损失小于当前的最小评估损失了
            min_val_loss = val_loss # 将最小评估损失替换为当前的损失
            best_model = copy.deepcopy(model) # 设置当前模型为目前最好的模型

        tqdm.write('Epoch {:03d} train_loss {:.5f} val_loss {:.5f}'.format(epoch, np.mean(train_loss), val_loss))

    torch.save(best_model.state_dict(), "model/cnn.pkl") # 保存状态字典


def test(): # 测试阶段
    Dtr, Val, Dte = load_data() # 将之前处理好的数据下载下来
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 如果可以用gpu的话，就用gpu运行否则用cpu
    model = cnn().to(device) # 设置模型
    model.load_state_dict(torch.load("model/cnn.pkl"), False) # 下载训练好的状态字典
    model.eval() # 将模型设置为评估模式
    total = 0 # 总测试数
    current = 0 # 测试成功的个数
    for (data, target) in Dte: # 在测试集中进行测试
        data, target = data.to(device), target.to(device)
        outputs = model(data) # 获得测试结果
        predicted = torch.max(outputs.data, 1)[1].data # 获得预测结果
        total += target.size(0) # 测试数+1
        current += (predicted == target).sum() # 如果测试结果等于预测结果，就+1

    print('Accuracy:%f%%' % (100 * current / total)) # 输出准确率


if __name__ == '__main__':
    train()
    test()
