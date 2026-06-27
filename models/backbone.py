"""
Backbone prunable (standalone) cho YOLOv1: ResNet18/34/50/101 + VGG16/19.

- Moi conv duoc boc PrunableConv (Conv+BN+Act + mask) => prune duoc.
- ResNet: an toan residual -> structured pruning CHI cat MID-channel cua block
  (BasicBlock: conv1; BottleNeck: conv1, conv2). Stem va output block GIU NGUYEN.
- VGG: tuyen tinh -> prune tu do moi conv.
- Width-based config (so kenh tuyet doi) cho surgery rebuild chinh xac:
    * ResNet: block_mids = list theo tung block; entry int (BasicBlock) hoac (mid1,mid2) (BottleNeck).
    * VGG:    widths = list so kenh moi conv.
"""
import torch
import torch.nn as nn

from .element import PrunableConv


# ---------------------------------------------------------------------------
# ResNet blocks
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, mid=None):
        super().__init__()
        mid = mid if mid is not None else out_channels
        self.conv1 = PrunableConv(in_channels, mid, kernel_size=3, stride=stride, padding=1, act="relu")
        self.conv2 = PrunableConv(mid, out_channels, kernel_size=3, stride=1, padding=1, act="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

    def get_prunable_layers(self, pruning_type="unstructured"):
        layers = list(self.conv1.get_prunable_layers(pruning_type))   # mid (an toan)
        if pruning_type == "unstructured":
            layers += self.conv2.get_prunable_layers(pruning_type)
            if isinstance(self.downsample, PrunableConv):
                layers += self.downsample.get_prunable_layers(pruning_type)
        return layers


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, mid1=None, mid2=None):
        super().__init__()
        mid1 = mid1 if mid1 is not None else out_channels
        mid2 = mid2 if mid2 is not None else out_channels
        out_full = out_channels * self.expansion
        self.conv1 = PrunableConv(in_channels, mid1, kernel_size=1, stride=1, padding=0, act="relu")
        self.conv2 = PrunableConv(mid1, mid2, kernel_size=3, stride=stride, padding=1, act="relu")
        self.conv3 = PrunableConv(mid2, out_full, kernel_size=1, stride=1, padding=0, act="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

    def get_prunable_layers(self, pruning_type="unstructured"):
        layers = list(self.conv1.get_prunable_layers(pruning_type))   # mid1
        layers += self.conv2.get_prunable_layers(pruning_type)        # mid2
        if pruning_type == "unstructured":
            layers += self.conv3.get_prunable_layers(pruning_type)
            if isinstance(self.downsample, PrunableConv):
                layers += self.downsample.get_prunable_layers(pruning_type)
        return layers


RESNET_CFG = {
    "resnet18": (BasicBlock, [2, 2, 2, 2], 512),
    "resnet34": (BasicBlock, [3, 4, 6, 3], 512),
    "resnet50": (BottleNeck, [3, 4, 6, 3], 2048),
    "resnet101": (BottleNeck, [3, 4, 23, 3], 2048),
}


class ResNet(nn.Module):
    def __init__(self, arch="resnet18", block_mids=None):
        super().__init__()
        block, layers, _ = RESNET_CFG[arch]
        self.arch = arch
        self.block = block
        self.in_channels = 64
        self._block_mids = block_mids   # list theo tung block (int hoac (mid1,mid2)); None -> dense
        self._mid_ptr = 0

        self.conv1 = PrunableConv(3, 64, kernel_size=7, stride=2, padding=3, act="relu")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _next_mid(self):
        if self._block_mids is None:
            self._mid_ptr += 1
            return None
        m = self._block_mids[self._mid_ptr] if self._mid_ptr < len(self._block_mids) else None
        self._mid_ptr += 1
        return m

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        out_full = out_channels * block.expansion
        if stride != 1 or self.in_channels != out_full:
            downsample = PrunableConv(self.in_channels, out_full, kernel_size=1,
                                      stride=stride, padding=0, act="identity")
        layers = []
        for b in range(num_block):
            mid = self._next_mid()
            blk_stride = stride if b == 0 else 1
            blk_down = downsample if b == 0 else None
            if block is BasicBlock:
                layers.append(block(self.in_channels, out_channels, blk_stride, blk_down, mid=mid))
            else:
                mid1 = mid2 = None
                if mid is not None:
                    mid1, mid2 = (mid if isinstance(mid, (tuple, list)) else (mid, mid))
                layers.append(block(self.in_channels, out_channels, blk_stride, blk_down,
                                    mid1=mid1, mid2=mid2))
            self.in_channels = out_full
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def get_prunable_layers(self, pruning_type="unstructured"):
        convs = []
        if pruning_type == "unstructured":
            convs += self.conv1.get_prunable_layers(pruning_type)   # stem: unstruct an toan
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for blk in stage:
                convs += blk.get_prunable_layers(pruning_type)
        return convs


# ---------------------------------------------------------------------------
# VGG
# ---------------------------------------------------------------------------
VGG_CFG = {
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
              512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
              512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, arch="vgg16", widths=None):
        super().__init__()
        cfg = VGG_CFG[arch]
        self.arch = arch

        layers = []
        c_in = 3
        ci = 0
        self.out_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                c_out = int(widths[ci]) if widths is not None else v
                layers.append(PrunableConv(c_in, c_out, kernel_size=3, stride=1, padding=1, act="relu"))
                c_in = c_out
                self.out_channels = c_out
                ci += 1
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

    def get_prunable_layers(self, pruning_type="unstructured"):
        convs = []
        for m in self.features:
            if isinstance(m, PrunableConv):
                convs += m.get_prunable_layers(pruning_type)
        return convs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_backbone(arch_name="resnet18", backbone_cfg=None, pretrained=False):
    """
    backbone_cfg:
      - ResNet: block_mids (list theo block) hoac None (dense).
      - VGG: widths (list theo conv) hoac None (dense).
    pretrained: nap ImageNet weights (CHI khi dense, backbone_cfg=None).
    Tra ve (backbone, feat_dims).
    """
    arch_name = arch_name.lower()
    if arch_name in RESNET_CFG:
        feat_dims = RESNET_CFG[arch_name][2]
        model = ResNet(arch_name, block_mids=backbone_cfg)
        if pretrained and backbone_cfg is None:
            load_imagenet_pretrained(model, arch_name)
        return model, feat_dims
    if arch_name in VGG_CFG:
        model = VGG(arch_name, widths=backbone_cfg)
        if pretrained and backbone_cfg is None:
            load_imagenet_pretrained(model, arch_name)
        return model, model.out_channels
    raise ValueError(f"Unsupported backbone: {arch_name}. "
                     f"Choose from {list(RESNET_CFG) + list(VGG_CFG)}")


# ---------------------------------------------------------------------------
# ImageNet-pretrained loading: remap theo TEN (resnet) / thu tu conv (vgg)
# ---------------------------------------------------------------------------
_TV_PRETRAINED = {
    "resnet18":  ("resnet18",  "ResNet18_Weights"),
    "resnet34":  ("resnet34",  "ResNet34_Weights"),
    "resnet50":  ("resnet50",  "ResNet50_Weights"),
    "resnet101": ("resnet101", "ResNet101_Weights"),
    "vgg16":     ("vgg16_bn",  "VGG16_BN_Weights"),   # HybridVGG co BN -> dung ban _bn
    "vgg19":     ("vgg19_bn",  "VGG19_BN_Weights"),
}


def _build_tv(arch):
    import torchvision
    fn_name, w_name = _TV_PRETRAINED[arch]
    weights = getattr(torchvision.models, w_name).DEFAULT
    return getattr(torchvision.models, fn_name)(weights=weights).eval()


def _remap_resnet_key(k):
    """tv key -> my key (PrunableConv boc conv+bn). None = bo qua (fc...)."""
    if k.startswith("conv1."):
        return "conv1.conv." + k[len("conv1."):]
    if k.startswith("bn1."):
        return "conv1.bn." + k[len("bn1."):]
    if k.startswith("layer"):
        for n in (1, 2, 3):
            if f".conv{n}." in k:
                return k.replace(f".conv{n}.", f".conv{n}.conv.")
            if f".bn{n}." in k:
                return k.replace(f".bn{n}.", f".conv{n}.bn.")
        if ".downsample.0." in k:
            return k.replace(".downsample.0.", ".downsample.conv.")
        if ".downsample.1." in k:
            return k.replace(".downsample.1.", ".downsample.bn.")
    return None


@torch.no_grad()
def _verify_backbone(model, tv, arch, input_size=224, atol=1e-3):
    model.eval()
    x = torch.randn(1, 3, input_size, input_size)
    my_out = model(x)
    if arch.startswith("resnet"):
        t = tv.maxpool(tv.relu(tv.bn1(tv.conv1(x))))
        t = tv.layer4(tv.layer3(tv.layer2(tv.layer1(t))))
        tv_out = t
    else:
        tv_out = tv.features(x)
    diff = (my_out - tv_out).abs().max().item()
    return torch.allclose(my_out, tv_out, atol=atol), diff


@torch.no_grad()
def load_imagenet_pretrained(model, arch, verify=True, verbose=True):
    arch = arch.lower()
    if arch not in _TV_PRETRAINED:
        raise ValueError(f"No pretrained mapping for {arch}")
    tv = _build_tv(arch)
    my_sd = model.state_dict()

    if arch.startswith("resnet"):
        new = {}
        for k, v in tv.state_dict().items():
            nk = _remap_resnet_key(k)
            if nk is None or nk not in my_sd:
                continue
            if my_sd[nk].shape != v.shape:
                raise RuntimeError(f"[pretrained] shape mismatch {k}->{nk}: "
                                   f"{tuple(v.shape)} vs {tuple(my_sd[nk].shape)}")
            new[nk] = v
        model.load_state_dict(new, strict=False)
        n_loaded = len(new)
    else:  # vgg (theo thu tu conv — chuoi tuyen tinh nen khong nhap nhang)
        tv_convs = [m for m in tv.features if isinstance(m, nn.Conv2d)]
        tv_bns = [m for m in tv.features if isinstance(m, nn.BatchNorm2d)]
        my_pc = [m for m in model.features if isinstance(m, PrunableConv)]
        if not (len(my_pc) == len(tv_convs) == len(tv_bns)):
            raise RuntimeError(f"[pretrained] VGG count mismatch: mine={len(my_pc)} "
                               f"tv_conv={len(tv_convs)} tv_bn={len(tv_bns)}")
        n_loaded = 0
        for pc, cv, bn in zip(my_pc, tv_convs, tv_bns):
            if pc.conv.weight.shape != cv.weight.shape:
                raise RuntimeError(f"[pretrained] VGG shape mismatch "
                                   f"{tuple(pc.conv.weight.shape)} vs {tuple(cv.weight.shape)}")
            pc.conv.weight.data.copy_(cv.weight.data)
            pc.bn.weight.data.copy_(bn.weight.data)
            pc.bn.bias.data.copy_(bn.bias.data)
            pc.bn.running_mean.data.copy_(bn.running_mean.data)
            pc.bn.running_var.data.copy_(bn.running_var.data)
            n_loaded += 1

    if verify:
        ok, diff = _verify_backbone(model, tv, arch)
        if not ok:
            raise RuntimeError(f"[pretrained] FORWARD MISMATCH (max diff={diff:.2e}) "
                               f"-> remap SAI, dung lai!")
        if verbose:
            print(f"[pretrained] {arch}: loaded {n_loaded} tensors, "
                  f"forward-verify OK (max diff={diff:.2e})")
    elif verbose:
        print(f"[pretrained] {arch}: loaded {n_loaded} tensors (no verify)")
    return model
