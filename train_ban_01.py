# model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from PIL import Image
from PIL import ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# display images
from torchvision import utils
import matplotlib.pyplot as plt


# utils
import numpy as np
import time
import copy

# Data preprocessing
trans_train = transforms.Compose([transforms.Resize((600,600)),
                            transforms.RandomCrop(512),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(degrees=(-90, 90)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
trans_val = transforms.Compose([transforms.Resize((512,512)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
trainset = torchvision.datasets.ImageFolder(root='E:/kim/abc/dataset/image/Training', transform=trans_train)
validationset = torchvision.datasets.ImageFolder(root='E:/kim/abc/dataset/image/Validation', transform=trans_val)

train_loader = DataLoader(trainset,
                         batch_size=4,
                         shuffle=True,
                         num_workers=6
                         )
val_loader = DataLoader(validationset,
                       batch_size=1,
                       shuffle=False,
                       num_workers=3
                       )
train_acces = []
train_losses = []
val_accse = []
val_losses = []
best_loss = 1000000000000
num = 0

device_txt = 'cuda:1'
device = torch.device(device_txt if torch.cuda.is_available() else "cpu")

print("--------------------------------------------------")

from models import EfficientNet_b4
from Custom_CosineAnnealingWarmRestarts import CosineAnnealingWarmUpRestarts

if __name__ == '__main__':

    torch.multiprocessing.freeze_support()
    model = EfficientNet_b4()
    model.to(device)

    epochs = 500

    # optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-12, momentum=0.9, weight_decay=1e-4)
    # scheduler
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=1e-3, T_up=10, gamma=0.5)
    print("********** GPU : ", device)

    best_model_wts = copy.deepcopy(model.state_dict())

    # train
    for epoch in range(0, 500):

        train_loss = 0
        train_num = 0

        if num > 10000000:
            num = 0

        now = time.time()

        for train_x, train_y in train_loader:

            num += 1

            model.train()
            train_x, train_y = train_x.to(device), train_y.to(device).long()

            optimizer.zero_grad()
            pred = model(train_x)
            loss = criterion(pred, train_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_num += 1

            if num % 1000 == 0:
                scheduler.step()


        train_avg_loss = train_loss / train_num
        print("Epoch : {}, Train_Avg_loss : {}".format(epoch, train_avg_loss))
        train_losses.append(train_avg_loss)

        # validation data check
        val_loss = 0
        val_num = 0

        for valid_x, valid_y in val_loader:
            with torch.no_grad():
                model.eval()
                valid_x, valid_y = valid_x.to(device), valid_y.to(device).long()
                pred = model(valid_x)
                loss = criterion(pred, valid_y)

            val_loss += loss.item()
            val_num += 1

        val_avg_loss = val_loss / val_num
        print("Epoch : {}, Val_Avg_loss : {}".format(epoch, val_avg_loss))
        val_losses.append(val_avg_loss)

        plt.plot(val_losses, label='val_loss')
        plt.plot(train_losses, label='train_loss')
        plt.xlabel('effi-b04_00')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

        savepath = 'E:/kim/abc/model/effi-b04_00/model_{}_LOSS_{}.pth'

        if best_loss > val_avg_loss:
            print("@@@@@@@@@@@ SAVE MODEL @@@@@@@@@@@")
            best_loss = val_avg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), savepath.format(epoch, best_loss))

        print(time.time() - now)