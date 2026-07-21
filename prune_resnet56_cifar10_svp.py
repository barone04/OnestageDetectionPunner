"""
prune_resnet56_cifar10_svp.py — Rebuild TRUNG THANH pipeline SLIMING / SVP-main
cho ResNet-56 / CIFAR-10, theo DUNG code released (TRAIN-FROM-SCRATCH).

Y tuong (bam sat SVP-main):
  1. Load pretrained ResNet-56 (baseline ~93.26%, file resnet_56.pt) CHI de tinh
     singular values cua tung mid-conv (conv1 moi block) prunable (KHONG copy weight).
  2. GAM (Greedy Addition Method) phan bo global so kenh MID giu lai theo --target-rate
     -> ra so kenh mid per-block. SVP KHONG release mapping ResNet (adapt_channel
     format chong index) -> ta build MID-ONLY residual-safe qua mid_channel_override
     (overall_channel giu = full, downsample/residual an toan), do MACs (thop) de dose.
  3. Build pruned resnet56 FRESH init (kaiming) va TRAIN FROM SCRATCH.
     -> SVP released trains from scratch; GEM (pruning/svp.py::select_filters_gem_*)
        chi dung khi copy+finetune — KHONG dung o day de bam code goc.

Hyperparams train-from-scratch theo SVP (paper Table 2 + code train.py/utils.py):
  epochs 300, lr 0.1, batch 128, SGD momentum 0.9, weight_decay 0.005,
  label_smoothing 0.1, mixup_alpha 0.2, cutmix_alpha 1.0,
  warmup 5 epoch LinearLR(start_factor=0.01) -> CosineAnnealingLR(eta_min=0),
  STEP PER-ITERATION (khop SVP train.py eras).

Data: torchvision.datasets.CIFAR10(root=--data-dir, download=True) — giong SVP data.py,
ap DUNG augment SVP data.py (ImageNet-stats Normalize + TrivialAugmentWide +
RandomErasing + Resize32) va mixup/cutmix trong collate_fn (port tu SVP utils.py).

Vi du:
  python prune_resnet56_cifar10_svp.py --pretrain resnet_56.pt --target-rate 0.5 --wandb

Import GAM tu pruning.svp (ban rebuild GAM cua minh):
  from pruning.svp import compute_singular_values, find_optimal_channels_gam
"""
import os
import math
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision
import torchvision.transforms as T

from pruning.svp import compute_singular_values, find_optimal_channels_gam


# ===========================================================================
# ResNet-56 CIFAR-10 — Option A (He/CORING/NORTON style: stem conv1/bn1, shortcut
# zero-pad KHONG param) de KHOP checkpoint SLIMING release (cifar10_resnet56_*.pt).
# GAM cap so kenh mid per-block qua mid_channel_override (overall giu full).
# ===========================================================================
def adapt_channel(compress_rates, layers):
    """
    Computes the overall and mid-channel sizes based on compression rates.
    PORT nguyen tu SVP-main/models/resnet.py.
    """
    num_stages = len(layers)

    if num_stages == 3:  # Resnet-CIFAR variant
        stage_out_channel = (
            [16] + [16] * layers[0] + [32] * layers[1] + [64] * layers[2]
        )
    elif num_stages == 4:  # Resnet-34 variant
        stage_out_channel = (
            [64]
            + [64] * layers[0]
            + [128] * layers[1]
            + [256] * layers[2]
            + [512] * layers[3]
        )
    else:
        raise ValueError("Unsupported number of stages. Expected 3 or 4.")

    stage_oup_cprate = [compress_rates[0]]
    for i in range(len(layers)):
        stage_oup_cprate += [compress_rates[i + 1]] * layers[i]
    stage_oup_cprate += [0.0] * layers[-1]
    mid_cprate = compress_rates[1:]

    overall_channel = []
    mid_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0:
            overall_channel += [int(stage_out_channel[i] * (1 - stage_oup_cprate[i]))]
        else:
            overall_channel += [int(stage_out_channel[i] * (1 - stage_oup_cprate[i]))]
            mid_channel += [int(stage_out_channel[i] * (1 - mid_cprate[i - 1]))]

    return overall_channel, mid_channel


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    """Option A (He CIFAR): shortcut = zero-pad KHONG param — khop ckpt SLIMING/CORING/NORTON."""

    expansion: int = 1

    def __init__(self, midplanes, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            pad = (planes - inplanes) // 2
            if stride != 1:
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, pad, planes - inplanes - pad), "constant", 0))
            else:
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x, (0, 0, 0, 0, pad, planes - inplanes - pad), "constant", 0))

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu2(out)


class ResNet(nn.Module):
    def __init__(
        self, block, layers, compress_rate, num_classes, width=16,
        mid_channel_override=None,
    ):
        super().__init__()
        # MID-ONLY: GAM cap so kenh mid per-block qua mid_channel_override, GIU
        # overall_channel = full (adapt_channel voi cprate 0). Shortcut Option A
        # (zero-pad, KHONG param) -> khop ckpt SLIMING release + CORING/NORTON.
        if mid_channel_override is not None:
            full_overall, _ = adapt_channel([0.0] * 100, layers)
            self.overall_channel = full_overall
            self.mid_channel = list(mid_channel_override)
        else:
            self.overall_channel, self.mid_channel = adapt_channel(compress_rate, layers)

        self.layer_num = 0
        self.conv1 = nn.Conv2d(
            3, self.overall_channel[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.overall_channel[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer_num = 1
        self.layer1 = self._make_layer(block, layers[0], stride=1)
        self.layer2 = self._make_layer(block, layers[1], stride=2)
        self.layer3 = self._make_layer(block, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.overall_channel[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, blocks, stride):
        layers = [block(self.mid_channel[self.layer_num - 1],
                        self.overall_channel[self.layer_num - 1],
                        self.overall_channel[self.layer_num], stride)]
        self.layer_num += 1
        for _ in range(1, blocks):
            layers.append(block(self.mid_channel[self.layer_num - 1],
                                self.overall_channel[self.layer_num - 1],
                                self.overall_channel[self.layer_num]))
            self.layer_num += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet56(compress_rate, num_classes, width=16, mid_channel_override=None):
    return ResNet(
        BasicBlock, [9, 9, 9], compress_rate, num_classes, width,
        mid_channel_override=mid_channel_override,
    )


# ===========================================================================
# mixup / cutmix — PORT nguyen tu SVP-main/utils.py (RandomMixup, RandomCutmix)
# ===========================================================================
class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup. PORT tu SVP utils.py."""

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0,
                 inplace: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix. PORT tu SVP utils.py."""

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0,
                 inplace: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        _, H, W = torchvision.transforms.functional.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target


def get_mixupcutmix(mixup_alpha, cutmix_alpha, num_classes):
    """PORT tu SVP data.py::get_mixupcutmix — RandomChoice(mixup, cutmix)."""
    return torchvision.transforms.RandomChoice(
        [
            RandomMixup(num_classes, p=1.0, alpha=mixup_alpha),
            RandomCutmix(num_classes, p=1.0, alpha=cutmix_alpha),
        ]
    )


# ===========================================================================
# Data — nguon = torchvision.datasets.CIFAR10 (auto-download), GIONG SVP data.py.
# mixup/cutmix ap trong collate_fn (giong SVP data.py::collate_fn ->
# mixupcutmix(*default_collate)).
#
# KHOP SVP data.py::get_transforms (doc that, thu tu transform giu nguyen):
#   - Normalize dung ImageNet stats mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
#     (SVP dung ImageNet stats du la CIFAR).
#   - Train: RandomHorizontalFlip, RandomCrop(32,pad4), TrivialAugmentWide,
#            ToTensor, Normalize, Resize(32, antialias=True), RandomErasing(0.1).
#   - Test:  ToTensor, Normalize, Resize(32, antialias=True).
# ===========================================================================
def cifar10_loaders(data_dir, batch_size, num_classes, mixup_alpha, cutmix_alpha, workers=4):
    """Nguon = torchvision.datasets.CIFAR10/100(root=data_dir, download=True) — GIONG SVP.
    Augment SVP-style + mixup/cutmix trong collate (KHOP SVP data.py goc)."""
    # CIFAR-10 stats (giong CORING/NORTON) — khop ckpt SLIMING Option-A -> eval baseline dung.
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    normalize = T.Normalize(mean, std)
    tr_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        normalize,
        T.Resize(32, antialias=True),
        T.RandomErasing(0.1),
    ])
    te_tf = T.Compose([
        T.ToTensor(),
        normalize,
        T.Resize(32, antialias=True),
    ])
    tvdset = torchvision.datasets.CIFAR10 if num_classes == 10 else torchvision.datasets.CIFAR100
    tr = tvdset(root=data_dir, train=True, download=True, transform=tr_tf)
    te = tvdset(root=data_dir, train=False, download=True, transform=te_tf)

    mixupcutmix = get_mixupcutmix(mixup_alpha, cutmix_alpha, num_classes)

    def collate_fn(batch):
        return mixupcutmix(*default_collate(batch))

    train_loader = DataLoader(
        tr, batch_size=batch_size, shuffle=True, num_workers=workers,
        drop_last=True, pin_memory=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        te, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,
    )
    return train_loader, test_loader


# ===========================================================================
# GAM allocation: load pretrained resnet56 CHI de tinh singular values -> GAM
# -> ra so kenh mid giu lai cho tung block (27 block).
#
# SVP KHONG release mapping ResNet; adapt_channel format bi chong index (dung
# chung compress_rate cho ca mid VA stage-output residual). -> Ta dung MID-ONLY
# residual-safe + GAM cap so kenh mid (mid_channel_override cho ResNet), do MACs
# de dose target_rate. Overall_channel giu = full model nen residual an toan.
# ===========================================================================
def _load_pretrained_baseline(pretrain_path, num_classes, device):
    """Build full resnet56 ([0.]*100) va load pretrained state_dict."""
    origin = resnet56([0.0] * 100, num_classes=num_classes).to(device)
    ck = torch.load(pretrain_path, map_location="cpu")
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    origin.load_state_dict(sd)
    return origin


def _block_mid_convs(model):
    """List conv1 (mid conv) cua tung block, thu tu layer1..layer3 (27 conv)."""
    convs = []
    for layer in (model.layer1, model.layer2, model.layer3):
        for block in layer:
            convs.append(block.conv1)
    return convs


def compute_compress_rate_gam(model, target_rate):
    """
    Chay GAM tren mid-conv (conv1) cua tung block -> so kenh mid giu lai per-block.

    Args:
        model: full resnet56 baseline (da load pretrained).
        target_rate: global compress rate (ti le kenh mid BI CAT).

    Returns:
        (keep list = so kenh mid giu per-block, orig_channels list)
        -> dung THANG keep lam mid_channel_override khi build pruned model
           (KHONG qua compress_rate broadcast loi cua adapt_channel).
    """
    mid_convs = _block_mid_convs(model)  # 27 conv (mid)
    weights = [c.weight.data.detach().cpu() for c in mid_convs]

    singular_values = [compute_singular_values(w) for w in weights]
    original_channels = [w.size(0) for w in weights]

    total_channels = int(np.sum(original_channels))
    total_to_keep = int((1.0 - target_rate) * total_channels)
    total_to_keep = max(total_to_keep, len(mid_convs))  # >=1 kenh/layer

    keep = find_optimal_channels_gam(
        singular_values, total_to_keep, original_channels
    )
    return keep, original_channels


def measure_macs_params(model, device, input_size=(1, 3, 32, 32)):
    """Do MACs + Params bang thop (input (1,3,32,32)). Tra (macs, params) hoac
    (None, None) neu thop chua cai — KHONG crash."""
    try:
        from thop import profile
    except ImportError:
        print("[WARN] thop chua cai (pip install thop) -> bo qua do MACs. "
              "Params van do bang params_count().")
        return None, None
    was_training = model.training
    model.eval()
    x = torch.randn(*input_size, device=device)
    with torch.no_grad():
        macs, params = profile(model, inputs=(x,), verbose=False)
    if was_training:
        model.train()
    return macs, params


# ===========================================================================
# Train-from-scratch loop — warmup(5ep, LinearLR start_factor=0.01) -> Cosine.
# Loss: soft-target CE (mixup/cutmix tra one-hot mix) voi label_smoothing.
# ===========================================================================
def soft_cross_entropy(outputs, soft_targets, label_smoothing=0.1):
    """CE voi soft targets (mixup/cutmix). Ap label smoothing thu cong vi target
    da la phan phoi one-hot mix (khong dung nn.CrossEntropyLoss class-index)."""
    num_classes = outputs.size(1)
    if label_smoothing > 0:
        soft_targets = soft_targets * (1.0 - label_smoothing) + label_smoothing / num_classes
    log_probs = torch.log_softmax(outputs, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


@torch.no_grad()
def validate(model, loader, device):
    """Tra (acc%, val_loss) — val_loss dung CrossEntropy hard-label (test khong mixup)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return 100.0 * correct / total, loss_sum / max(total, 1)


def params_count(m):
    return sum(p.numel() for p in m.parameters())


def _load_dotenv(path=".env"):
    """Tu nap .env (KEY=VALUE) vao os.environ neu bien chua co. Khong can python-dotenv.
    -> WANDB_API_KEY/WANDB_PROJECT/WANDB_ENTITY tu san sang, khong phai export tay."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    _load_dotenv()   # tu nap .env -> tu bat wandb neu co WANDB_API_KEY (khong can dan tay)
    ap = argparse.ArgumentParser(
        description="SVP/SLIMING rebuild — prune ResNet-56/CIFAR-10 (GAM + train-from-scratch)"
    )
    ap.add_argument("--pretrain", required=True,
                    help="resnet_56.pt (baseline ~93.26%%) — CHI de tinh GAM singular values")
    ap.add_argument("--target-rate", default=0.5, type=float,
                    help="global compress rate (ti le kenh mid bi cat). Default 0.5")
    ap.add_argument("--num-classes", default=10, type=int)
    # Hyperparams train-from-scratch DUNG SVP (paper Table 2 + code)
    ap.add_argument("--epochs", default=300, type=int)
    ap.add_argument("--lr", default=0.1, type=float)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--momentum", default=0.9, type=float)
    ap.add_argument("--weight-decay", default=0.005, type=float)
    ap.add_argument("--label-smoothing", default=0.1, type=float)
    ap.add_argument("--mixup-alpha", default=0.2, type=float)
    ap.add_argument("--cutmix-alpha", default=1.0, type=float)
    ap.add_argument("--warmup-epochs", default=5, type=int)
    ap.add_argument("--eta-min", default=0.0, type=float)
    ap.add_argument("--data-dir", default="./data", help="root tai CIFAR-10 (torchvision auto-download)")
    ap.add_argument("--workers", default=4, type=int)
    ap.add_argument("--output-dir", default="./output/resnet56_sliming")
    ap.add_argument("--no-wandb", action="store_true",
                    help="tat wandb (MAC DINH tu bat neu co WANDB_API_KEY trong .env/env)")
    ap.add_argument("--wandb-name", default="resnet56-cifar10-sliming")
    ap.add_argument("--wandb-project", default="resnet56-cifar10-sliming")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader = cifar10_loaders(
        args.data_dir, args.batch_size, args.num_classes,
        args.mixup_alpha, args.cutmix_alpha, args.workers,
    )

    # --- 1) Baseline pretrained CHI de tinh GAM config (KHONG copy weight) ---
    origin = _load_pretrained_baseline(args.pretrain, args.num_classes, device)
    p0 = params_count(origin)
    base_acc, _ = validate(origin, val_loader, device)
    print(f"Baseline ResNet-56 (chi de GAM) | params={p0/1e6:.3f}M | acc={base_acc:.2f}%")

    # Do MACs/Params baseline (dense) truoc khi prune.
    macs0, thop_p0 = measure_macs_params(origin, device)

    keep, orig_ch = compute_compress_rate_gam(origin, args.target_rate)
    kept, tot = sum(keep), sum(orig_ch)
    print(f"[GAM] target_rate={args.target_rate:.3f} -> mid kept {kept}/{tot} "
          f"({kept/tot*100:.1f}%) across {len(keep)} blocks")
    print(f"[GAM] mid_channel_override (keep per-block): {keep}")

    del origin  # KHONG copy weight — giai phong baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- 2) Build pruned resnet56 FRESH init & TRAIN FROM SCRATCH ---
    # MID-ONLY residual-safe: overall_channel = full, chi doi mid_channel = keep
    # (GAM). compress_rate placeholder [0.]*100 khong dung khi co override.
    # SVP released trains from scratch; GEM (pruning/svp.py) chi dung khi copy+finetune
    # -> KHONG dung o day de bam code goc.
    net = resnet56(
        [0.0] * 100, num_classes=args.num_classes, mid_channel_override=keep,
    ).to(device)
    p1 = params_count(net)
    reduction = (1 - p1 / p0) * 100
    print(f"Pruned ResNet-56 (fresh init) | params={p1/1e6:.3f}M "
          f"(-{reduction:.1f}% vs baseline {p0/1e6:.3f}M)")

    # Do MACs/Params pruned + in % giam (dose --target-rate toi ~58% MACs, paper Table 4).
    macs1, thop_p1 = measure_macs_params(net, device)
    if macs0 is not None and macs1 is not None:
        macs_drop = (1 - macs1 / macs0) * 100
        params_drop = (1 - thop_p1 / thop_p0) * 100
        print(f"[thop] MACs: {macs0/1e6:.2f}M -> {macs1/1e6:.2f}M (-{macs_drop:.1f}%) | "
              f"Params: {thop_p0/1e6:.3f}M -> {thop_p1/1e6:.3f}M (-{params_drop:.1f}%)")
        print("[thop] Dose --target-rate sao cho MACs giam ~58% (paper Table 4).")
    else:
        macs_drop = params_drop = None

    wb = None
    use_wandb = (not args.no_wandb) and bool(os.getenv("WANDB_API_KEY"))
    if use_wandb:
        import wandb   # WANDB_API_KEY tu .env -> wandb.init tu login
        wb = wandb.init(name=args.wandb_name, project=args.wandb_project, config=vars(args))
        wb.summary["baseline_acc"] = base_acc
        wb.summary["pruned_params_M"] = p1 / 1e6
        wb.summary["params_reduction_pct"] = reduction
        if macs1 is not None:
            wb.summary["pruned_macs_M"] = macs1 / 1e6
            wb.summary["macs_reduction_pct"] = macs_drop

    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # KHOP SVP train.py (eras): warmup era = LinearLR(start_factor=0.01) chay
    # `warmup_epochs * len(train_loader)` ITERATIONS, roi main era = Cosine chay
    # `epochs * len(train_loader)` iterations. STEP PER-ITERATION (goi trong vong
    # batch, KHONG per-epoch). Gop bang SequentialLR, milestone tinh theo iteration.
    iters_per_epoch = len(train_loader)
    warmup_iters = args.warmup_epochs * iters_per_epoch
    cosine_iters = max(1, args.epochs * iters_per_epoch)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_iters
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_iters, eta_min=args.eta_min
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters]
    )

    best = 0.0
    save_path = os.path.join(args.output_dir, "model_best.pth")
    net.train()
    for ep in range(args.epochs):
        loss_sum = n = 0
        for x, y in train_loader:
            # mixup/cutmix da tra soft target (one-hot mix) qua collate_fn.
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = net(x)
            loss = soft_cross_entropy(out, y, args.label_smoothing)
            loss.backward()
            optimizer.step()
            scheduler.step()  # PER-ITERATION (khop SVP train.py)
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = loss_sum / max(n, 1)
        acc, val_loss = validate(net, val_loader, device)
        if acc > best:
            best = acc
            torch.save({"state_dict": net.state_dict(), "acc": best,
                        "mid_channels": keep}, save_path)
        lr = optimizer.param_groups[0]["lr"]
        if wb is not None:
            wb.log({"train_loss": train_loss, "val_acc": acc, "best_acc": best,
                    "val_loss": val_loss, "lr": lr})
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep+1}/{args.epochs}] loss={train_loss:.3f} val_loss={val_loss:.3f} "
                  f"acc={acc:.2f}% best={best:.2f}% lr={lr:.4e}")

    if wb is not None:
        wb.summary["final_acc"] = best
        if os.path.exists(save_path):
            wb.save(save_path)   # upload weight len wandb -> hien tren run (Kaggle/bat ky dau)
        wb.finish()

    print(f"\nDONE. Best acc={best:.2f}% | pruned params={p1/1e6:.3f}M "
          f"(-{reduction:.1f}%) | saved: {save_path}")
    print("Note: paper Table 4 ResNet-56 -> 94.15% @ 58% MACs.")


if __name__ == "__main__":
    main()
