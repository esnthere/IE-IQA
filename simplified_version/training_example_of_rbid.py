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


def train(model,model2, train_loader, optimizer, epoch, device, all_train_loss):
    model.train()
    model.apply(fix_bn)
    st = time.time()
    op=[]
    tg=[]
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        data /= 255
        data[:, 0] -= 0.485
        data[:, 1] -= 0.456
        data[:, 2] -= 0.406
        data[:, 0] /= 0.229
        data[:, 1] /= 0.224
        data[:, 2] /= 0.225
        with torch.no_grad():
            model2.eval()
            model2.eval()
            fts = model2.module.extract_features(data)
        fts = fts.to(device)
        optimizer.zero_grad()
        output = model(data, None, fts)
        loss = F.mse_loss(output, target)

        all_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        op = np.concatenate((op, output[:, 0].detach().cpu().numpy()))
        tg = np.concatenate((tg, target[:, 0].detach().cpu().numpy()))
        # if batch_idx % 100 == 0:
        #     print(time.time() - st)
        #     print('Train Epoch: {} [({:.0f}%)]\tLoss: {:.4f}'.format(
        #         epoch, 100. * batch_idx / len(train_loader), loss.item()))
    print('Train Epoch:',epoch,  'ALL Pearson:', pd.Series(op).corr(pd.Series(tg), method="pearson"))
    print('Train Epoch:',epoch,   'ALL Spearman:', pd.Series(op).corr(pd.Series(tg), method="spearman"))
    return all_train_loss


def test(model,model2, test_loader, epoch, device, all_test_loss):
    model.eval()
    test_loss = 0
    pearson = 0
    spearman = 0
    op = []
    op2=[]
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
            p1 = pd.Series(output[:, 0].cpu()).corr(pd.Series(target[:, 0].cpu()), method="pearson")
            s1 = pd.Series(output[:, 0].cpu()).corr(pd.Series(target[:, 0].cpu()), method="spearman")
            pearson += p1
            spearman += s1
            # if batch_idx % 100 == 0:
            #     print('Test Epoch:{} [({:.0f}%)]\t Loss: {:.4f}  Pearson:{:.4f} Spearman:{:.4f}'.format(
            #         epoch, 100. * batch_idx / len(test_loader), loss.item(), p1, s1))

        test_loss /= (batch_idx + 1)
        pearson /= (batch_idx + 1)
        spearman /= (batch_idx + 1)
        # print('Test : Loss:{:.4f} '.format(test_loss))
    print('Test Epoch:',epoch,   'ALL Pearson:', pd.Series(op).corr(pd.Series(tg), method="pearson"))
    print('Test Epoch:',epoch,   'ALL Spearman:', pd.Series(op).corr(pd.Series(tg), method="spearman"))
    return all_test_loss, pd.Series(op).corr(pd.Series(tg), method="pearson"), pd.Series(op).corr(pd.Series(tg), method="spearman")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    device = torch.device("cuda")
    global  ns
    ns=64

    all_data = sio.loadmat('E:\Database\RBID\\rbid_244.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = Y *0.8 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = Ytest *0.8 + 1
    del all_data
    plcc=[]
    srcc=[]
    for i in range(10):
        print('Split:', i)
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

        train_dataset = Mydataset(X, Y)
        test_dataset = Mydataset(Xtest, Ytest)

        model = EfficientNet.from_name('efficientnet-b0').to(device)

        model_dict = model.state_dict()  # 读出搭建的网络的参数，以便后边更新之后初始化

        # 去除不属于model_dict的键值
        pretrained_dict = torch.load('efficientnet-b0.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 更新现有的model_dict的值
        model_dict.update(pretrained_dict)

        # 加载模型需要的参数
        model.load_state_dict(model_dict)

        model2 = EfficientNet2.from_name('efficientnet-b0').to(device)
        model_dict = model2.state_dict()
        pretrained_dict = torch.load('efficientnet-b0.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model2.load_state_dict(model_dict)
        model2 = nn.DataParallel(model2)

        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc1.parameters():
            param.requires_grad = True
        for param in model.fc2.parameters():
            param.requires_grad = True
        for param in model.fc3.parameters():
            param.requires_grad = True

        model = nn.DataParallel(model)


        # model.load_state_dict(torch.load("Koniq__imagenet_fts_cls1_1.pt"))
        ###################################################################


        min_loss = 1e8
        lr = 0.01
        weight_decay = 1e-4
        batch_size = 64*2
        epochs = 2000
        num_workers_train = 0
        num_workers_test = 0

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        train_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_train,pin_memory=True)
        test_loader = DataLoaderX(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,pin_memory=True)

        all_train_loss = []
        all_test_loss = []
        all_test_loss, _,_ = test(model,model2, test_loader, -1, device, all_test_loss)
        ct = 0
        lr=0.01
        max_plsp=-2

        for epoch in range(epochs):
            print(lr)
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                  weight_decay=weight_decay)
            ct += 1
            start = time.time()
            all_train_loss = train(model,model2, train_loader, optimizer, epoch, device, all_train_loss)
            print(time.time() - start)
            all_test_loss, plsp,_ = test(model, model2,test_loader, epoch, device, all_test_loss)
            print("time:", time.time() - start)
            if epoch==20:
                for param in model.parameters():
                    param.requires_grad = True
                lr=0.001

            if max_plsp < plsp:
                save_nm = "bid_imgnetfts_split"+str(i)+".pt"
                max_plsp = plsp
                torch.save(model.state_dict(), save_nm)
                ct=0

            if  epoch ==40:
                lr = 0.01
            if  epoch ==60:
                lr = 0.03
                ct=1

            if ct>20 and epoch > 60:
                model.load_state_dict(torch.load(save_nm))
                lr *= 0.1
                ct=0
                if lr<5e-5:
                    model.load_state_dict(torch.load(save_nm))
                    _,pl,sp=test(model, model2, test_loader, epoch, device, all_test_loss)
                    plcc.append(pl)
                    srcc.append(sp)

                    break
    print('Spilit:',i,'End!','All PLCC:',plcc,'All SRCC:',srcc)

if __name__ == '__main__':
    main()

