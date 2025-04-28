import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict


class MNIST_LeNet(nn.Module):
    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x


class FashionMNIST_LeNet(nn.Module):
    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 7 * 7, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        return x


class MyLinear(nn.Linear):
    def __init__(self, input_dim, output_dim, bias=True, device=None):
        super(MyLinear, self).__init__(input_dim, output_dim, bias, device)

    def forward(self, input):
        return input @ self.weight.T - self.bias


class SCos(nn.Module):
    def __init__(self, input_dim, learn_sigm):
        super(SCos, self).__init__()

        self.learn_sigm, initval = learn_sigm
        self.coef = torch.tensor(input_dim ** (-0.5))
        self.sigm = nn.Parameter(torch.ones(1) * initval, requires_grad=self.learn_sigm)

    def forward(self, input):
        inputSigm = input * self.sigm
        return self.coef * torch.hstack([torch.sin(inputSigm), torch.cos(inputSigm)])

    def extra_repr(self) -> str:
        return f"sigm_len={1}, learn_sigm={self.learn_sigm}"

class Model(nn.Module):
    def __init__(self, useBackbone, head_layer, output_dim):
        super(Model, self).__init__()
        num_layers = len(head_layer)
        self.dropout = nn.Dropout(0.5)

        if useBackbone == "ResNet50":
            self.backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2")
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, head_layer[0])
        elif useBackbone == "ResNet18":
            self.backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, head_layer[0])
        else:
            raise ValueError("Cannot find the backbone")

        self.hidlayer = nn.Sequential(OrderedDict({"hidlayer" + str(idx + 1): nn.Sequential(nn.Linear(head_layer[idx], head_layer[idx + 1], bias=False),
                                                                                            # self.dropout,
                                                                                            nn.ReLU()) for idx in range(num_layers - 1)}))
        self.outlayer = nn.Linear(head_layer[-1], output_dim, bias=True)

        print(f"The network with {useBackbone} backbone, {num_layers - 1} regular hid.layer is initialized.")
        for name, para in self.named_parameters():
            if name[:8] != "backbone":
                print({name: [para.shape, para.requires_grad]})

    def forward(self, x):
        x = self.backbone(x)
        x = self.hidlayer(x)

        return self.outlayer(x)
