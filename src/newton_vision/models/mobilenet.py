"""MobileNetV2 Derin Evrişimsel Sinir Ağı Mimarisi (newton_vision.models.mobilenet)."""

from __future__ import annotations
import numpy as np
import newton_vision as nv
from newton_vision.module import Module


class InvertedResidual(Module):
    """MobileNetV2 Inverted Residual Block (1x1 Conv -> Depthwise 3x3 Conv -> 1x1 Linear Conv)."""
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))

        layers_list = []
        if expand_ratio != 1:
            layers_list.extend([
                nv.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nv.BatchNorm2d(hidden_dim),
                nv.ReLU()
            ])

        layers_list.extend([
            nv.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nv.BatchNorm2d(hidden_dim),
            nv.ReLU(),
            nv.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nv.BatchNorm2d(out_channels)
        ])

        self.conv = nv.Sequential(*layers_list)

    def forward(self, x: nv.Tensor) -> nv.Tensor:
        out = self.conv(x)
        if self.use_res_connect:
            return x + out
        return out


class MobileNetV2(Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.cfgs = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = 32
        self.conv1 = nv.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False)
        self.bn1 = nv.BatchNorm2d(input_channel)
        self.relu1 = nv.ReLU()

        features_list = []
        for t, c, n, s in self.cfgs:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features_list.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features = nv.Sequential(*features_list)

        self.conv_last = nv.Conv2d(input_channel, 1280, 1, bias=False)
        self.bn_last = nv.BatchNorm2d(1280)
        self.relu_last = nv.ReLU()

        self.avgpool = nv.AvgPool2d(7, stride=1)
        self.classifier = nv.Linear(1280, num_classes)

    def forward(self, x: nv.Tensor) -> nv.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.features(out)

        out = self.relu_last(self.bn_last(self.conv_last(out)))
        out = self.avgpool(out)

        N = out.data.shape[0]
        flat = out.reshape(N, 1280)

        return self.classifier(flat)


def mobilenet_v2(num_classes: int = 1000) -> MobileNetV2:
    return MobileNetV2(num_classes=num_classes)
