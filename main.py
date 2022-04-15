import torch  # for set model status(cpu, gpu), optimizer, torch.no_grad()
import gc
import os
import time
import warnings
from model import * 
import numpy as np
import pandas as pd  # for make some dataframe of train,test loss. it will be printed at last
import torch.nn as nn  # for define loss function
from tqdm import tqdm  # for checking how many time on 'for' statement processing.
from dataloader import ad_dataset  # from dataset.py
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
import torch.nn.functional as F
from torchvision.models import resnet50
from torchsummary import summary
import torch.optim as optim

gc.collect()

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss




def train(trn_loader, model, device,  optimizer):
    model.train()
    model.to(device)
    trn_loss = 0
    train_mse = 0
    best_loss = np.inf

    for (data, target) in tqdm(trn_loader):  # i means how many
        optimizer.zero_grad()  # pytorch has gradient before nodes
        data = data.to(device)
        output = model(data)  # input data in model
        output = output.type(torch.FloatTensor)
        target = target.to(device)
        target = target.type(torch.FloatTensor)
        target = target.reshape(target.shape)

        trn_loss = focal_loss(target, output,alpha=0.25,gamma=2)  # cost fcn is Binary_Cross_entropy
        trn_loss.backward()  # backpropagation
        optimizer.step()  # training model

        trn_loss += trn_loss.item()
        train_mse += torch.mean(torch.abs(output - target) ** 2).item()
        del data, target, output
        gc.collect()

    trn_loss /= len(trn_loader)
    train_mse /= len(trn_loader)
    print(f'Train Loss:{trn_loss:.5f} | MSE:{train_mse:.5f}')
    return trn_loss, train_mse




def test(tst_loader, model, device):
    model.eval()
    model.to(device)
    tst_loss = 0
    tst_mse = 0
    best_loss = np.inf

    with torch.no_grad():
        for data, target in tqdm(tst_loader):
            data = data.to(device)
            output = model(data)
            output = output.type(torch.FloatTensor)
            target = target.to(device)
            target = target.type(torch.FloatTensor)
            target = target.reshape(output.shape)
            tst_loss = focal_loss(target, output,alpha=0.25,gamma=2)

            tst_loss += tst_loss.item()
            tst_mse += torch.mean(torch.abs(output - target)**2).item()
            del data, target, output
            gc.collect()

        tst_loss /= len(tst_loader)
        tst_mse /= len(tst_loader)
        print(f'Val Loss:{tst_loss:.5f} | MSE:{tst_mse:.5f}')
    return tst_loss, tst_mse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--arch', type=str, default='Multi_Res50')
    args = parser.parse_args()
    print(args)
    main(args)


