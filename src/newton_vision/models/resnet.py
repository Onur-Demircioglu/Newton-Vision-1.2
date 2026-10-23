"""ResNet-18 Derin Evrişimsel Sinir Ağı Mimarisi (newton_vision.models.resnet)."""

from __future__ import annotations
import numpy as np
import newton_vision as nv
from newton_vision.module import Module


class Downsample(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nv.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.bn = nv.BatchNorm2d(out_channels)

    def forward(self, x: nv.Tensor) -> nv.Tensor:
        return self.bn(self.conv(x))


class ResidualBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Module | None = None):
        super().__init__()
        self.conv1 = nv.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nv.BatchNorm2d(out_channels)
        self.relu1 = nv.ReLU()

        self.conv2 = nv.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nv.BatchNorm2d(out_channels)
        self.relu2 = nv.ReLU()

        self.downsample = downsample

    def forward(self, x: nv.Tensor) -> nv.Tensor:
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu2(out)


class ResNet18(Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nv.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nv.BatchNorm2d(64)
        self.relu = nv.ReLU()
        self.maxpool = nv.MaxPool2d(kernel_size=3, stride=2, padding=1)  # torchvision ResNet-18 ile aynı

        self.layer1 = nv.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))

        down2 = Downsample(64, 128, stride=2)
        self.layer2 = nv.Sequential(ResidualBlock(64, 128, stride=2, downsample=down2), ResidualBlock(128, 128))

        down3 = Downsample(128, 256, stride=2)
        self.layer3 = nv.Sequential(ResidualBlock(128, 256, stride=2, downsample=down3), ResidualBlock(256, 256))

        down4 = Downsample(256, 512, stride=2)
        self.layer4 = nv.Sequential(ResidualBlock(256, 512, stride=2, downsample=down4), ResidualBlock(512, 512))

        self.avgpool = nv.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nv.Linear(512, num_classes)

    def forward(self, x: nv.Tensor) -> nv.Tensor:
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        N = out.data.shape[0]
        flat = out.reshape(N, 512)

        return self.fc(flat)


def resnet18(num_classes: int = 1000) -> ResNet18:
    return ResNet18(num_classes=num_classes)
