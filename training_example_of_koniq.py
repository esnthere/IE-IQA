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
from imgaug import augmenters as iaa
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import time
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from skimage import io
import os
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import torchvision.models as models
from my_efficientnet_ie_feature import EfficientNet as EfficientNet2
from my_efficientnet import EfficientNet
from prefetch_generator import BackgroundGenerator
import matplotlib.pyplot as plt
import lmdb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


class Mydataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = torch.FloatTensor(labels)

    def __getitem__(self, index):

        img = self.imgs[index]

        return torch.from_numpy(img),  self.labels[index]

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
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        torch.random.manual_seed(len(train_loader) * epoch + batch_idx)
        rd_ps = torch.randint(20, (3,))
        data = data[:, :, rd_ps[0]:rd_ps[0] + 224, rd_ps[1]:rd_ps[1] + 224]
        if rd_ps[1] < 10:
            data = torch.flip(data, dims=[3])
        data=data.float()
        data/=255
        data[:, 0] -= 0.485
        data[:, 1] -= 0.456
        data[:, 2] -= 0.406
        data[:, 0] /= 0.229
        data[:, 1] /= 0.224
        data[:, 2] /= 0.225
        with torch.no_grad():
            model2.eval()
            fts = model2.module.extract_features(data)
            mfts = torch.mean(fts, (2, 3))
            fts2 = torch.zeros_like(mfts)[:,:ns]
            output=model2.module._fc(mfts)
            five_maxcls = torch.argsort(output)[:, -1:]
            for i in range(five_maxcls.shape[1]):
                maxfts = torch.sort(torch.argsort(torch.abs(mfts * model2.module._fc.weight.data[five_maxcls[:, i]]))[:, -ns:],1)[0]
                for j in range(five_maxcls.shape[0]):
                    fts2[j] = mfts[j, maxfts[j]]

        optimizer.zero_grad()
        output, output2 = model(data, fts2)
        loss1 = F.mse_loss(output, target)
        loss2 = F.mse_loss(output2, target)
        loss = loss1 + 0 * loss2
        all_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % 100== 0:
            print(time.time() - st)
            print('Train Epoch: {} [({:.0f}%)]\tLoss: {:.4f}Loss1: {:.4f}Loss2: {:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item(), loss1.item(), loss2.item()))
    return all_train_loss


def test(model,model2, test_loader, epoch, device, all_test_loss):
    model.eval()
    test_loss = 0

    op = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, target ) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data[:, :,10:10+ 224, 10:10+ 224]

            data = data.float()
            data /= 255
            data[:,0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:,0] /= 0.229
            data[:,1] /= 0.224
            data[:,2] /= 0.225
            model2.eval()
            fts = model2.module.extract_features(data)
            mfts = torch.mean(fts, (2, 3))
            fts2 = torch.zeros_like(mfts)[:,:ns]
            output=model2.module._fc(mfts)
            five_maxcls = torch.argsort(output)[:, -1:]
            for i in range(five_maxcls.shape[1]):
                maxfts = torch.sort(torch.argsort(torch.abs(mfts * model2.module._fc.weight.data[five_maxcls[:, i]]))[:, -ns:],1)[0]
                for j in range(five_maxcls.shape[0]):
                    fts2[j] = mfts[j, maxfts[j]]

            output, output2 = model(data, fts2)
            loss1 = F.mse_loss(output, target)
            loss2 = F.mse_loss(output2, target)
            loss = loss1 +0* loss2
            all_test_loss.append(loss)
            test_loss += loss
            op = np.concatenate((op, output[:, 0].cpu().numpy()))
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))


            if batch_idx % 100 == 0:
                print('Test Epoch:{} [({:.0f}%)]\t Loss: {:.4f}  Loss1: {:.4f}  Loss2: {:.4f} '.format(
                        epoch, 100. * batch_idx / len(test_loader), loss.item(), loss1.item(), loss2.item()))

    test_loss /= (batch_idx + 1)

    print('Test : Loss:{:.4f} '.format(test_loss))
    print( 'ALL Pearson:', pd.Series(op).corr(pd.Series(tg), method="pearson"))
    print( 'ALL Spearman:', pd.Series(op).corr(pd.Series(tg), method="spearman"))

    return all_test_loss, pd.Series(op).corr(pd.Series(tg), method="pearson")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    device = torch.device("cuda")
    global  ns
    ns=64
    model = EfficientNet.from_name('efficientnet-b0',adp_num=ns).to(device)


    model_dict = model.state_dict()  # 读出搭建的网络的参数，以便后边更新之后初始化

    # 去除不属于model_dict的键值
    pretrained_dict = torch.load('efficientnet-b0.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict }

    # 更新现有的model_dict的值
    model_dict.update(pretrained_dict)
    # 加载模型需要的参数
    model.load_state_dict(model_dict)

    model2 =   EfficientNet2.from_name('efficientnet-b0').to(device)
    model_dict = model2.state_dict()  # 读出搭建的网络的参数，以便后边更新之后初始化
    pretrained_dict = torch.load('efficientnet-b0.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict }
    model_dict.update(pretrained_dict)
    model2.load_state_dict(model_dict)


    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model2 = nn.DataParallel(model2, device_ids=[0, 1,2,3])

    for param in model.parameters():
        param.requires_grad = False

    for param in model.module.fc1.parameters():
        param.requires_grad = True
    for param in model.module.fc2.parameters():
        param.requires_grad = True
    for param in model.module.fc3.parameters():
        param.requires_grad = True
    for param in model.module.fc12.parameters():
        param.requires_grad = True
    for param in model.module.fc22.parameters():
        param.requires_grad = True
    for param in model.module.fc32.parameters():
        param.requires_grad = True


    # model.load_state_dict(torch.load("Koniq__imagenet_fts_cls1_1.pt"))
    ###################################################################

    all_data = sio.loadmat('E:\Database\IQA Database\## KonIQ-10k Image Database\Koniq_244.mat')
    X = all_data['X']
    Y= all_data['Y'].transpose(1, 0)
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    del all_data
    train_dataset = Mydataset( X, Y)

    test_dataset = Mydataset(Xtest, Ytest)




    min_loss = 1e8
    lr = 0.01
    weight_decay = 1e-5
    batch_size = 64*4
    epochs = 2000
    num_workers_train = 0
    num_workers_test = 0

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_train,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,pin_memory=True)

    all_train_loss = []
    all_test_loss = []
    all_test_loss, test_loss = test(model,model2, test_loader, -1, device, all_test_loss)
    ct = 0
    lr=0.01
    max_plsp=-2

    for epoch in range(epochs):
        print(lr)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,  weight_decay=weight_decay)
        ct += 1
        start = time.time()
        all_train_loss = train(model,model2, train_loader, optimizer, epoch, device, all_train_loss)
        print(time.time() - start)
        all_test_loss, plsp = test(model, model2,test_loader, epoch, device, all_test_loss)
        print("time:", time.time() - start)
        if epoch==10:
            for param in model.parameters():
                param.requires_grad = True
            lr=0.001

        if max_plsp < plsp:
            save_nm = "Koniq_imgnetfts.pt"
            max_plsp = plsp
            torch.save(model.state_dict(), save_nm)
            ct=0

        if  epoch ==20:
            lr = 0.01
        if  epoch ==30:
            lr = 0.03
            ct=1

        if ct>5 and epoch > 30:
            model.load_state_dict(torch.load(save_nm))
            lr *= 0.1
            ct=0
            if lr<6e-5:
                break

if __name__ == '__main__':
    main()

