import torch  # for using torch.sigmoid
import torch.nn as nn  # for using nn.Module
from torchsummary import summary  # for checking amount of model parameter
import torch.nn.init as init

class Multi_Res50(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.classifier = nn.Linear(1000, 30)

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)

        return torch.sigmoid(x)


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.layer1 = nn.Sequential(
            torch.nn.Linear(256*256*3, 1024, bias=True),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            torch.nn.Linear(1024, 512, bias=True),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            torch.nn.Linear(512, 256, bias=True),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            torch.nn.Linear(256, 128, bias=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            torch.nn.Linear(128, 64, bias=True),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.layer6 = nn.Sequential(
            torch.nn.Linear(64, 30, bias=True)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x_out = self.layer1(x)
        x_out = self.layer2(x_out)
        x_out = self.layer3(x_out)
        x_out = self.layer4(x_out)
        x_out = self.layer5(x_out)
        x_out = self.layer6(x_out)
        return torch.sigmoid(x_out)

