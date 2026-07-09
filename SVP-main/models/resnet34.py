from typing import List
import torch
import torch.nn as nn
from .resnet import adapt_channel, BasicBlock


__all__ = [
    "resnet18",
    "resnet34",
]


class ResNet(nn.Module):
    """
    Residual Neural Network (ResNet) architecture for ImageNet classification.

    Args:
        block (nn.Module): The residual block module used in the network.
        layers (list): A list of integers representing the number of residual blocks in each layer.
        compress_rate (list): A list of integers representing the compress rate for each layer.
        num_classes (int): The number of output classes.
        width (int): The width multiplier for the network.

    Attributes:
        inplanes (int): The number of input channels to the network.
        dilation (int): The dilation rate for the convolutional filters.
        layer_num (int): The current layer number.
        embed (nn.Sequential): The embedding layers of the network.
        layer1 (nn.Sequential): The first layer of residual blocks.
        layer2 (nn.Sequential): The second layer of residual blocks.
        layer3 (nn.Sequential): The third layer of residual blocks.
        layer4 (nn.Sequential): The fourth layer of residual blocks.
        fc (nn.Linear): The fully connected layer for classification.

    """

    def __init__(
        self,
        block,
        layers,
        compress_rate,
        num_classes,
        width,
    ):
        super().__init__()

        self.inplanes = width
        self.dilation = 1
        self.overall_channel, self.mid_channel = adapt_channel(compress_rate, layers)

        self.layer_num = 0
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.overall_channel[self.layer_num],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(self.overall_channel[self.layer_num])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_num += 1

        self.layer1 = self._make_layer(block, width, layers[0], stride=1)
        self.layer2 = self._make_layer(
            block,
            width * 2,
            layers[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            block,
            width * 4,
            layers[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            block,
            width * 8,
            layers[3],
            stride=2,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        last_out_channels = self.layer4[-1].conv2.out_channels
        self.fc = nn.Linear(last_out_channels * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != self.overall_channel[self.layer_num]:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.overall_channel[self.layer_num - 1],
                    out_channels=self.overall_channel[self.layer_num],
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.overall_channel[self.layer_num]),
            )

        layers = []
        layers.append(
            block(
                midplanes=self.mid_channel[self.layer_num - 1],
                inplanes=self.overall_channel[self.layer_num - 1],
                planes=self.overall_channel[self.layer_num],
                stride=stride,
                downsample=downsample,
            )
        )
        self.layer_num += 1

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    midplanes=self.mid_channel[self.layer_num - 1],
                    inplanes=self.overall_channel[self.layer_num - 1],
                    planes=self.overall_channel[self.layer_num],
                    stride=1,
                )
            )
            self.layer_num += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(
    compress_rate: List[float] = [0.0] * 9,
    num_classes: int = 1000,
    width: int = 64,
):
    return ResNet(BasicBlock, [2, 2, 2, 2], compress_rate, num_classes, width)


def resnet34(
    compress_rate: List[float] = [0.0] * 17,
    num_classes: int = 1000,
    width: int = 64,
):
    return ResNet(BasicBlock, [3, 4, 6, 3], compress_rate, num_classes, width)
