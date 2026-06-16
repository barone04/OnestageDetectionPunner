"""
Neck (ConvBlock) prunable cho YOLOv1 — theo kien truc reference: 5 conv (1x1,3x3,1x1,3x3,1x1).

Component-aware: khi backbone nhe di, neck tro thanh bottleneck. Toan bo 5 conv la
PrunableConv => structured filter pruning truc tiep (chuoi tuyen tinh).
Width-based: widths = [w0,w1,w2,w3,w4] (so kenh tuyet doi) cho surgery rebuild.
"""
import torch.nn as nn

from .element import PrunableConv


class ConvBlock(nn.Module):
    KS = [1, 3, 1, 3, 1]
    PAD = [0, 1, 0, 1, 0]

    def __init__(self, in_channels, out_channels, widths=None):
        super().__init__()
        base = [out_channels, out_channels * 2, out_channels, out_channels * 2, out_channels]
        chans = [int(w) for w in widths] if widths is not None else base

        convs = []
        c_in = in_channels
        for i in range(5):
            convs.append(PrunableConv(c_in, chans[i], kernel_size=self.KS[i],
                                      padding=self.PAD[i], act="leaky"))
            c_in = chans[i]
        self.convs = nn.Sequential(*convs)
        self.out_channels = chans[-1]

    def forward(self, x):
        return self.convs(x)

    def get_prunable_layers(self, pruning_type="unstructured"):
        convs = []
        for m in self.convs:
            if isinstance(m, PrunableConv):
                convs += m.get_prunable_layers(pruning_type)
        return convs
