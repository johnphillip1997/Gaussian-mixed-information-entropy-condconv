import torch.nn as nn
import math
import torch
import torchvision.models as torchmodels
import re
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.utils as torchutils
from torch.nn import init, Parameter


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise_conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch
        )
        self.depthwise_conv2 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch
        )
        self.depthwise_conv3 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=5,
            stride=stride,
            padding=2,
            groups=in_ch
        )
        self.depthwise_conv4 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=7,
            stride=stride,
            padding=3,
            groups=in_ch
        )
        self.pointwise_conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=int(in_ch / 4),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.pointwise_conv2 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=int(in_ch / 4),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.pointwise_conv3 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=int(in_ch / 4),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.pointwise_conv4 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=int(in_ch / 4),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

        self.pointwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        out1 = self.pointwise_conv1(self.depthwise_conv1(x))
        out2 = self.pointwise_conv2(self.depthwise_conv2(x))
        out3 = self.pointwise_conv3(self.depthwise_conv3(x))
        out4 = self.pointwise_conv4(self.depthwise_conv4(x))
        out = torch.cat([out1, out2, out3, out4], 1)
        out = self.pointwise_conv(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out

class DownsampleB(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.max = nn.MaxPool2d(stride)
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.max(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)



