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



class Mydataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = torch.FloatTensor(labels)

    def __getitem__(self, index):
        if self.imgs[0].shape[2]==244:
            img=self.imgs[index, :, 10:10 + 224, 10:10 + 224]
        else:
            img = self.imgs[index]

        return torch.from_numpy(img),  self.labels[index]

    def __len__(self):
        return (self.imgs).shape[0]





def test(model, model2,test_loader, epoch, device, all_test_loss):
    model.eval()
    model2.eval()

    op = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data=data.float()
            data/=255
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225
            fts = model2.module.extract_features(data)
            mfts = torch.mean(fts, (2, 3))
            fts2 = torch.zeros_like(mfts)[:, :ns]
            output = model2.module._fc(mfts)
            five_maxcls = torch.argsort(output)[:, -1:]
            for i in range(five_maxcls.shape[1]):
                maxfts = torch.sort(torch.argsort(torch.abs(mfts * model2.module._fc.weight.data[five_maxcls[:, i]]))[:, -ns:],1)[0]
                for j in range(five_maxcls.shape[0]):
                    fts2[j] = mfts[j, maxfts[j]]

            output, _ = model(data, fts2)

            op = np.concatenate((op, output[:, 0].cpu().numpy()))
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))


    print('ALL Pearson:', pd.Series(op).corr(pd.Series(tg), method="pearson"))
    print( 'ALL Spearman:', pd.Series(op).corr(pd.Series(tg), method="spearman"))
    return 0




def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    global  ns
    ns=64
    model = EfficientNet.from_name('efficientnet-b0',adp_num=ns).to(device)
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load("IEIQA.pt"))

    model2 = EfficientNet2.from_name('efficientnet-b0').to(device)
    model_dict = model2.state_dict()
    pretrained_dict = torch.load('efficientnet-b0.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model2.load_state_dict(model_dict)
    model2 = nn.DataParallel(model2, device_ids=[0])

    batch_size = 64
    num_workers_test = 0

    #########################################################

    all_data = sio.loadmat('E:\Database\IQA Database\## KonIQ-10k Image Database\Koniq_224.mat')
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    del all_data

    test_dataset = Mydataset(Xtest, Ytest)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,
                              pin_memory=True)

    print("koniq Test Results:")
    all_test_loss = []
    test(model,model2, test_loader, -1, device, all_test_loss)

    #####################################    
    all_data = sio.loadmat('E:\Database\LIVEW\livew_224.mat')

    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)
    Y = Y / 25 + 1

    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    Ytest = Ytest / 25 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X,Xtest),0), np.concatenate((Y,Ytest),0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,
                              pin_memory=True)
    print("Livew Test Results:")

    all_test_loss = []
    test(model,model2, test_loader, -1, device, all_test_loss)
    ######################################################

    all_data = sio.loadmat('E:\Database\CID2013\cid_224.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = (Y ) / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = (Ytest ) / 25 + 1
    del all_data

    test_dataset = Mydataset(np.concatenate((X,Xtest),0), np.concatenate((Y,Ytest),0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,
                              pin_memory=True)

    all_test_loss = []
    print("CID Test Results:")
    test(model,model2, test_loader, -1, device, all_test_loss)
    #######################################################

    all_data = sio.loadmat('E:\Database\RBID\\rbid_224.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = Y * 0.8 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = Ytest * 0.8 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), 0), np.concatenate((Y, Ytest), 0))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,
                             pin_memory=True)

    all_test_loss = []
    print("RBID Test Results:")
    test(model, model2, test_loader, -1, device, all_test_loss)
    #######################################################

    all_data = sio.loadmat('E:\Database\SPAQ\spaq_224.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = Y.reshape(Y.shape[1], 1)
    Y = Y / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = Ytest.reshape(Ytest.shape[1], 1)
    Ytest = Ytest / 25 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), 0), np.concatenate((Y, Ytest), 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,
                              pin_memory=True)

    all_test_loss = []
    print("SPAQ Test Results:")
    test(model,model2, test_loader, -1, device, all_test_loss)


if __name__ == '__main__':
    main()

