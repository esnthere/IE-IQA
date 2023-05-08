import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import math
from scipy import io as sio
import torch.utils.data
import torchvision.models as models
import pandas as pd
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
import os
import time
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import torchvision.models as models
from my_efficientnet import EfficientNet
from my_efficientnet_ie_feature import EfficientNet as EfficientNet2
from prefetch_generator import BackgroundGenerator
import matplotlib.pyplot as plt


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Mydataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = torch.FloatTensor(labels)
    def __getitem__(self, index):
        return torch.from_numpy(self.imgs[index]), self.labels[index]
    def __len__(self):
        return (self.imgs).shape[0]


def detect_loss(pred,pred2, label,label2):
    loss1 = torch.mean(torch.pow(pred - label, 2)) #+ torch.mean(torch.abs(pred- label))
    T=5
    loss2=-torch.mean(torch.mean(F.softmax(label2/T)*F.softmax(pred2/T).log(),1))

    return loss1,loss2




def test(model,model2, test_loader, epoch, device, all_test_loss):
    model.eval()
    test_loss = 0
    pearson = 0
    spearman = 0
    op = []

    tg = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data.float()
            data /= 255
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225

            model2.eval()
            fts = model2.module.extract_features(data)
            fts = fts.to(device)
            output = model(data, None, fts)
            loss = F.mse_loss(output, target)
            all_test_loss.append(loss)
            test_loss += loss
            op = np.concatenate((op, output[:, 0].cpu().numpy()))
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))

    print('Test Epoch:',epoch,   'ALL Pearson:', pd.Series(op).corr(pd.Series(tg), method="pearson"))
    print('Test Epoch:',epoch,   'ALL Spearman:', pd.Series(op).corr(pd.Series(tg), method="spearman"))
    return all_test_loss, pd.Series(op).corr(pd.Series(tg), method="pearson"), pd.Series(op).corr(pd.Series(tg), method="spearman")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    device = torch.device("cuda")
    global  ns
    ns=64
    model = EfficientNet.from_name('efficientnet-b0').to(device)

    model_dict = model.state_dict()  # 读出搭建的网络的参数，以便后边更新之后初始化

    # 去除不属于model_dict的键值
    pretrained_dict = torch.load('efficientnet-b0.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 更新现有的model_dict的值
    model_dict.update(pretrained_dict)

    # 加载模型需要的参数
    model.load_state_dict(model_dict)
    model = nn.DataParallel(model)

    model2 = EfficientNet2.from_name('efficientnet-b0').to(device)
    model_dict = model2.state_dict()
    pretrained_dict = torch.load('efficientnet-b0.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model2.load_state_dict(model_dict)
    model2 = nn.DataParallel(model2)

    model.load_state_dict(torch.load('IEIQA_rbid.pt'))
##############################
    all_data = sio.loadmat('E:\Database\RBID\\rbid_244.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = Y * 0.8 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = Ytest * 0.8 + 1
    del all_data

    for i in range(2):
        if i > -0:
            X = np.concatenate((X, Xtest), axis=0)
            Y = np.concatenate((Y, Ytest), axis=0)
            ind = np.arange(0, X.shape[0])
            np.random.seed(i)
            np.random.shuffle(ind)

            Xtest = X[ind[int(len(ind) * 0.8):]]
            Ytest = Y[ind[int(len(ind) * 0.8):]]
            X = X[ind[:int(len(ind) * 0.8)]]
            Y = Y[ind[:int(len(ind) * 0.8)]]

    test_dataset = Mydataset(Xtest, Ytest)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0,pin_memory=True)
    print("RBID Test Results:")

    test(model, model2, test_loader, 0, device, [])





if __name__ == '__main__':
    main()

