"""
YOLOv1 detection head — 1x1 conv tao tensor du doan (obj + box + class) cho moi cell.

Output channels = (1 obj + 4 box) * 1 box + num_classes  (theo reference: 1 box / cell).
=> KHONG prune output (rang buoc grid/box/class). Chi input (= neck output) co the giam
khi neck bi prune (xu ly o surgery).
"""
import torch.nn as nn


class YoloHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_attributes = (1 + 4) * 1 + num_classes
        self.detect = nn.Conv2d(in_channels, self.num_attributes, kernel_size=1)

    def forward(self, x):
        return self.detect(x)
