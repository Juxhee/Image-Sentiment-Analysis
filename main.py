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




def main(args):
    start = time.time()
    model_path = '../results/'
    experiment_num = 'Resnet_imb'
    dir = "../ad_data"
    save_path = os.path.join(model_path, experiment_num)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        if len(os.listdir(save_path)) > 1:
            print('Create New Folder')
            raise ValueError
        else:
            pass

    os.environ["CUDA_VISIBLE_DEVICE"] = '1'


    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
        transforms.ToTensor()
    ])

    dataset = ad_dataset(data_dir=dir, transform=transform)

    test_split = .2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)


    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(dataset, batch_size=16, sampler=test_sampler, num_workers=0)
    
    # define device
    if args.device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # define model
    if args.arch == 'Multi_Res50':
        model = Multi_Res50()
    elif args.arch == 'DNN':
        model = DNN()
    
    # define optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5)

    # Training, Validate
    best_loss = np.inf
    for epoch in range(1, args.num_epochs + 1):
        print('{}Epoch'.format(epoch))
        train_loss, train_mse, train_acc = train(train_loader, model, device=device, optimizer=optimizer,criterion=criterion)
        val_loss, val_mse, val_acc = test(test_loader, model,criterion=criterion, device=device)
        scheduler.step(val_loss)
        # Save Models
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model, os.path.join(save_path, 'best_model.pth'))  # 전체 모델 저장
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model_state_dict.pth'))  # 모델 객체의 state_dict 저장

        if epoch == 60:
            torch.save(model, os.path.join(save_path, f'{epoch}epoch.pth'))
            torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}epoch_state_dict.pth'))
        write_logs(epoch, train_mse, val_mse, train_acc, val_acc, save_path)
    end = time.time()
    print(f'Total Process time:{(end - start) / 60:.3f}Minute')
    print(f'Best Epoch:{best_epoch} | MSE:{best_loss:.5f}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--arch', type=str, default='Multi_Res50')
    args = parser.parse_args()
    print(args)
    main(args)


