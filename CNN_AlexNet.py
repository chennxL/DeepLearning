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

from torchvision.models import alexnet # 导入alexnet模型
model = alexnet(pretrained=True) # 使用预训练好的模型
model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 16)) # alexnet最后输出是1000类，将他改成16类

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 如果可以用gpu就用，否则用cpu

def setup_seed(seed): #生成随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)

def get_val_loss(model, Val):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    val_loss = []
    for (data, target) in Val: # data是图片，target是图片的类别
        data, target = Variable(data).to(device), Variable(target.long()).to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss.append(loss.cpu().item())

    return np.mean(val_loss)


def train(model, Dtr, Val, path, lr):
    model = model.to(device)
    print('train...')
    epoch_num = 100 # 设置训练回合数为50（比较少，因为如果回合太大了要训练很久）
    best_model = None
    min_epochs = 5
    min_val_loss = 5
    optimizer = optim.Adam(model.parameters(), lr=lr) # 选取优化器为Adam
    criterion = nn.CrossEntropyLoss().to(device) # #计算交叉熵损失
    # criterion = nn.BCELoss().to(device)
    for epoch in tqdm(range(epoch_num), ascii=True):
        train_loss = [] # 训练损失
        for batch_idx, (data, target) in enumerate(Dtr, 0): # enumerate的功能是计数，batch_idx就是在记录当前取了多少数据，索引从0开始
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            # target = target.view(target.shape[0], -1)
            optimizer.zero_grad() # 梯度下降
            output = model(data) # 将data放到模型中进行计算，得到结果
            loss = criterion(output, target) # 计算损失
            loss.backward() # 反向传播计算得到每个参数的梯度值
            optimizer.step() # 梯度下降执行一步参数更新
            train_loss.append(loss.cpu().item()) # 添加损失结果
        # validation
        val_loss = get_val_loss(model, Val) # 得到评估的损失
        model.train() # 调整模型为训练模式（因为在上一步中设成评估模式了）
        if epoch + 1 > min_epochs and val_loss < min_val_loss: # 如果评估损失小于当前的最小评估损失了
            min_val_loss = val_loss # 将最小评估损失替换为当前的损失
            best_model = copy.deepcopy(model) # 设置当前模型为目前最好的模型

        tqdm.write('Epoch {:03d} train_loss {:.5f} val_loss {:.5f}'.format(epoch, np.mean(train_loss), val_loss))

    torch.save(best_model.state_dict(), path) # 保存状态字典

    return best_model # 返回结果最好的model


def test(model, Dte, path):
    model.load_state_dict(torch.load(path), False)  # 下载训练好的状态字典
    model = model.to(device)
    model.eval()  # 将模型设置为评估模式
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
    Dtr, Val, Dte = load_data()
    lr = 0.0001
    path = './data/net.pth' # 保存参数
    train(model, Dtr, Val, path, lr)
    test(model, Dte, path)
