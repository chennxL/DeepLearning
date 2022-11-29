from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


def Myloader(path):
    return Image.open(path).convert('RGB') # 读图像并将图像转换为三通道的

def init_process(path, lens, name, pre): # 得到图像的path和lebal
    data = []
    for i in range(lens): #lens就是这个类中图片的个数
        i = i+pre # i+pre获得当前图片的编号
        j=str(i).zfill(5) # zfill将图片名称设置为5位数字，右对齐，不足的左边补0
        data.append([path % j, name]) #添加当前图片的信息
    print(data)
    return data


class MyDataset(Dataset):
    def __init__(self, data, transform, loader): # 初始化
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item): # 返回图像和标签
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self): # 返回数据集长度
        return len(self.data)

def load_data():
    print('data processing...')
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3), # 对图片进行随机水平翻转
        transforms.RandomVerticalFlip(p=0.3), # 对图片进行随机竖直翻转，这两个翻转的目的是实现数据增强
        transforms.Resize((256, 256)), #对图片进行缩放
        transforms.ToTensor(),#转化为tensor格式
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 对图片进行标准化处理
    ])

    # 每一个path代表每一个类的图片的路径，data就是在process函数中处理的结果，得到的是路径和标签的一个list
    path1 = 'data/training_data/0/%s.jpg'
    data1 = init_process(path1, 451,0,0)

    path2 = 'data/training_data/1/%s.jpg'
    data2 = init_process(path2, 267,1,451)

    path3 = 'data/training_data/2/%s.jpg'
    data3 = init_process(path3, 831,2,718)

    path4 = 'data/training_data/3/%s.jpg'
    data4 = init_process(path4, 1038,3,1549)

    path5 = 'data/training_data/4/%s.jpg'
    data5 = init_process(path5, 546, 4, 2587)

    path6 = 'data/training_data/5/%s.jpg'
    data6 = init_process(path6, 255, 5, 3133)

    path7 = 'data/training_data/6/%s.jpg'
    data7 = init_process(path7,239, 6, 3388)

    path8 = 'data/training_data/7/%s.jpg'
    data8 = init_process(path8, 100, 7, 3627)

    path9 = 'data/training_data/8/%s.jpg'
    data9 = init_process(path9, 105, 8, 3727)

    path10 = 'data/training_data/9/%s.jpg'
    data10 = init_process(path10, 625, 9, 3832)

    path11 = 'data/training_data/10/%s.jpg'
    data11 = init_process(path11, 244, 10, 4484)

    path12 = 'data/training_data/11/%s.jpg'
    data12 = init_process(path12, 104, 11, 4728)

    path13 = 'data/training_data/12/%s.jpg'
    data13 = init_process(path13, 327, 12, 4832)

    path14 = 'data/training_data/13/%s.jpg'
    data14 = init_process(path14, 608, 13, 5159)

    path15 = 'data/training_data/14/%s.jpg'
    data15 = init_process(path15, 217, 14, 5767)

    path16 = 'data/training_data/15/%s.jpg'
    data16 = init_process(path16, 776, 15, 5984)

    # 总的data数
    data = data1 + data2 + data3 + data4 + data5 + data6 + data7 + data8 + data9 + data10 + data11 + data12 + data13 + data14 + data15 + data16# 1400
    # 将data随机打乱顺序
    np.random.shuffle(data)
    #将训练集、评估集、测试集分开，其中0-5000是训练集，5000-5800是评估集，5800以后的是测试集
    train_data, val_data, test_data = data[:5000], data[5000:5800], data[5800:]
    # 包装一下需要使用的数据，方便后续读取
    train_data = MyDataset(train_data, transform=transform, loader=Myloader)
    Dtr = DataLoader(dataset=train_data, batch_size=50, shuffle=True, num_workers=0)
    val_data = MyDataset(val_data, transform=transform, loader=Myloader)
    Val = DataLoader(dataset=val_data, batch_size=50, shuffle=True, num_workers=0)
    test_data = MyDataset(test_data, transform=transform, loader=Myloader)
    Dte = DataLoader(dataset=test_data, batch_size=50, shuffle=True, num_workers=0)

    return Dtr, Val, Dte