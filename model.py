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
            torch.nn.Linear(256*256*3, 256, bias=True),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            torch.nn.Linear(256, 64, bias=True),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            torch.nn.Linear(64, 30, bias=True)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x_out = self.layer1(x)
        x_out = self.layer2(x_out)
        x_out = self.layer3(x_out)
        return x_out

