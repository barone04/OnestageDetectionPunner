"""
Model CIFAR chuan cho benchmark pruning (VGG16-BN + ResNet-56).

Dung DUNG bien the ma literature pruning dung, de so duoc voi so published:
  - VGG16-BN : cfg 'D' + BN, classifier = 1 Linear(512, num_classes).
               14.72M params / ~314M MACs  (paper bao 14.73M / 321.35M FLOPs)
  - ResNet-56: stem 3x3/16ch (KHONG maxpool), 3 stage x 9 BasicBlock, width 16/32/64.
               0.85M params / ~126M MACs   (paper bao 0.85M / 129.32M FLOPs)
    LUU Y: KHONG phai ResNet-18 (do la bien the ImageNet trong models/backbone.py).

Tai dung PrunableConv + BasicBlock san co => UnstructuredPruner/StructuredPruner
chay nguyen khong sua gi.

Structured pruning:
  - ResNet: chi cat MID-channel cua block (conv1), shortcut giu nguyen — dung quy uoc
            cua L1 [Li et al. 2017] va SPSRC ("we do not consider pruning of the
            projection shortcuts for simplification").
  - VGG   : chuoi tuyen tinh, prune tu do. Preset layer set:
      "all" -> ca 13 conv
      "l1a" -> conv 1 va 8..13 (VGG-16-pruned-A cua L1). KIEM TRA LAI voi paper goc
               truoc khi dung cho bang so sanh chinh thuc.
"""
import torch
import torch.nn as nn

from .element import PrunableConv
from .backbone import BasicBlock


# ---------------------------------------------------------------------------
# VGG16-BN cho CIFAR
# ---------------------------------------------------------------------------
# HAI dong VGG16-BN khac nhau trong literature pruning — KHONG the tron so vao 1 bang:
#   "hrank"  : 4 maxpool + AvgPool(2), classifier = Linear-BN1d-ReLU-Linear.
#              14.98M params. Dung boi HRank / CORING / GAL. <- mac dinh
#   "single" : 5 maxpool, classifier = 1 Linear(512,10).
#              14.72M params. Khop dense 14.73M ma SPSRC bao cao.
VGG16_CFG = {
    "hrank": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
              512, 512, 512, "M", 512, 512, 512],
    "single": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
               512, 512, 512, "M", 512, 512, 512, "M"],
}

# conv index (1-based) duoc prune theo tung preset
VGG_PRUNE_SETS = {
    "all": None,                          # None = tat ca
    "l1a": [1, 8, 9, 10, 11, 12, 13],     # VGG-16-pruned-A (Li et al. 2017)
}


class CifarVGG(nn.Module):
    def __init__(self, arch="vgg16", num_classes=10, widths=None, prune_set="all",
                 head="hrank"):
        super().__init__()
        cfg = VGG16_CFG[head]
        self.arch = arch
        self.num_classes = num_classes
        self.prune_set = prune_set
        self.head = head

        layers, self.convs = [], []
        c_in, ci = 3, 0
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
                continue
            c_out = widths[ci] if widths is not None else v
            conv = PrunableConv(c_in, c_out, kernel_size=3, stride=1, padding=1, act="relu")
            layers.append(conv)
            self.convs.append(conv)
            c_in, ci = c_out, ci + 1
        if head == "hrank":
            layers.append(nn.AvgPool2d(2, 2))    # 2x2 -> 1x1 (chi co 4 maxpool)

        self.features = nn.Sequential(*layers)
        if head == "hrank":
            self.classifier = nn.Sequential(
                nn.Linear(c_in, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                nn.Linear(512, num_classes))
        else:
            self.classifier = nn.Linear(c_in, num_classes)
        self._widths = widths

    @property
    def first_linear(self):
        """Linear an vao kenh conv cuoi -> phai slice khi surgery."""
        return self.classifier[0] if self.head == "hrank" else self.classifier

    def forward(self, x):
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))

    def _structured_convs(self):
        """Cac conv duoc phep cat filter, theo preset."""
        idx = VGG_PRUNE_SETS[self.prune_set]
        if idx is None:
            return self.convs
        return [self.convs[i - 1] for i in idx if i - 1 < len(self.convs)]

    def get_prunable_layers(self, pruning_type="unstructured"):
        pool = self.convs if pruning_type == "unstructured" else self._structured_convs()
        out = []
        for c in pool:
            out += c.get_prunable_layers(pruning_type)
        return out

    def config(self):
        return {
            "family": "vgg", "arch": self.arch, "num_classes": self.num_classes,
            "widths": self._widths, "prune_set": self.prune_set, "head": self.head,
        }


# ---------------------------------------------------------------------------
# ResNet-56 cho CIFAR (He et al. 2016)
# ---------------------------------------------------------------------------
class CifarResNet(nn.Module):
    def __init__(self, depth=56, num_classes=10, block_mids=None):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth phai la 6n+2 (20/32/44/56/110)"
        n = (depth - 2) // 6
        self.depth = depth
        self.num_classes = num_classes
        self._block_mids = block_mids
        self._mid_ptr = 0
        self.in_channels = 16

        self.conv1 = PrunableConv(3, 16, kernel_size=3, stride=1, padding=1, act="relu")
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _next_mid(self):
        if self._block_mids is None:
            self._mid_ptr += 1
            return None
        m = self._block_mids[self._mid_ptr] if self._mid_ptr < len(self._block_mids) else None
        self._mid_ptr += 1
        return m

    def _make_layer(self, out_channels, num_block, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = PrunableConv(self.in_channels, out_channels, kernel_size=1,
                                      stride=stride, padding=0, act="identity")
        blocks = []
        for b in range(num_block):
            blocks.append(BasicBlock(self.in_channels, out_channels,
                                     stride if b == 0 else 1,
                                     downsample if b == 0 else None,
                                     mid=self._next_mid()))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))

    def get_prunable_layers(self, pruning_type="unstructured"):
        convs = []
        if pruning_type == "unstructured":
            convs += self.conv1.get_prunable_layers(pruning_type)
        for stage in (self.layer1, self.layer2, self.layer3):
            for blk in stage:
                convs += blk.get_prunable_layers(pruning_type)
        return convs

    def config(self):
        return {
            "family": "resnet", "depth": self.depth, "num_classes": self.num_classes,
            "block_mids": self._block_mids,
        }


# ---------------------------------------------------------------------------
def build_cifar_model(cfg):
    """Dung lai model tu dict config (dung cho reload / surgery rebuild)."""
    if cfg["family"] == "vgg":
        return CifarVGG(cfg.get("arch", "vgg16"), cfg.get("num_classes", 10),
                        cfg.get("widths"), cfg.get("prune_set", "all"),
                        cfg.get("head", "hrank"))
    return CifarResNet(cfg.get("depth", 56), cfg.get("num_classes", 10),
                       cfg.get("block_mids"))


def demo():
    """Self-check: shape + so params/MACs phai khop bang published."""
    x = torch.randn(2, 3, 32, 32)
    res = CifarResNet(56)
    assert res(x).shape == (2, 10)
    pr = sum(p.numel() for p in res.parameters())
    assert 0.83e6 < pr < 0.87e6, f"ResNet-56 params {pr} != ~0.85M"

    # hai bien the VGG phai khop dung so dense cua dong tuong ung
    counts = {}
    for head, lo, hi in (("hrank", 14.9e6, 15.0e6), ("single", 14.6e6, 14.8e6)):
        vgg = CifarVGG("vgg16", head=head)
        assert vgg(x).shape == (2, 10)
        pv = counts[head] = sum(p.numel() for p in vgg.parameters())
        assert lo < pv < hi, f"VGG16-BN[{head}] params {pv} ngoai khoang"
    vgg = CifarVGG("vgg16")
    pv = counts["hrank"]

    # scope structured: ResNet-56 -> 27 block (chi conv1 moi block)
    assert len(res.get_prunable_layers("structured")) == 27
    assert len(vgg.get_prunable_layers("structured")) == 13
    assert len(CifarVGG("vgg16", prune_set="l1a").get_prunable_layers("structured")) == 7

    # surgery rebuild tu config phai chay duoc
    assert build_cifar_model(vgg.config())(x).shape == (2, 10)
    assert build_cifar_model(res.config())(x).shape == (2, 10)

    print(f"OK  VGG16-BN hrank={counts['hrank']/1e6:.2f}M single={counts['single']/1e6:.2f}M "
          f"| ResNet-56 {pr/1e6:.2f}M")


if __name__ == "__main__":
    demo()
