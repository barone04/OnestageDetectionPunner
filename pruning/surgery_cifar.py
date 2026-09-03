"""
Model surgery cho model CIFAR (CifarVGG / CifarResNet) — song song voi surgery.py (YOLO).

Chuoi phu thuoc:
  VGG    : tuyen tinh. conv_k.in = kept(conv_{k-1}); classifier.in = kept(conv cuoi).
  ResNet : chi MID-channel cua block giam (conv1.out). conv2.out va shortcut GIU NGUYEN
           => in-channel cua moi block va fc khong doi.
"""
import os
import json

import torch

from models.cifar import CifarVGG, CifarResNet, build_cifar_model
from models.element import PrunableConv
from .surgery import _kept_idx, _copy_sliced


# ---------------------------------------------------------------------------
def _vgg_widths(model: CifarVGG):
    """So kenh con song moi conv. Conv khong nam trong prune set -> giu nguyen."""
    prunable = set(id(c) for c in model._structured_convs())
    return [int(_kept_idx(c).numel()) if id(c) in prunable else c.out_channels
            for c in model.convs]


def _resnet_mids(model: CifarResNet):
    return [int(_kept_idx(blk.conv1).numel())
            for stage in (model.layer1, model.layer2, model.layer3) for blk in stage]


# ---------------------------------------------------------------------------
@torch.no_grad()
def _surgery_vgg(src: CifarVGG, dst: CifarVGG):
    prunable = set(id(c) for c in src._structured_convs())
    in_idx = torch.arange(3)
    for s_conv, d_conv in zip(src.convs, dst.convs):
        out_idx = (_kept_idx(s_conv) if id(s_conv) in prunable
                   else torch.arange(s_conv.out_channels))
        _copy_sliced(d_conv, s_conv, in_idx, out_idx)
        in_idx = out_idx
    # classifier: feature map 1x1 -> cot cua Linear dau map 1-1 voi kenh conv cuoi
    dst.first_linear.weight.data.copy_(src.first_linear.weight.data[:, in_idx])
    dst.first_linear.bias.data.copy_(src.first_linear.bias.data)
    if src.head == "hrank":     # BN1d + Linear cuoi khong doi chieu
        for d, s in zip(dst.classifier[1:], src.classifier[1:]):
            if hasattr(s, "weight"):
                d.load_state_dict(s.state_dict())


@torch.no_grad()
def _surgery_resnet(src: CifarResNet, dst: CifarResNet):
    _copy_sliced(dst.conv1, src.conv1, torch.arange(3), torch.arange(src.conv1.out_channels))

    for s_stage, d_stage in zip((src.layer1, src.layer2, src.layer3),
                                (dst.layer1, dst.layer2, dst.layer3)):
        for s_blk, d_blk in zip(s_stage, d_stage):
            full_in = torch.arange(s_blk.conv1.in_channels)
            mid_idx = _kept_idx(s_blk.conv1)
            _copy_sliced(d_blk.conv1, s_blk.conv1, full_in, mid_idx)
            _copy_sliced(d_blk.conv2, s_blk.conv2, mid_idx,
                         torch.arange(s_blk.conv2.out_channels))
            # ShortcutA khong co tham so -> khong copy gi
            if isinstance(s_blk.downsample, PrunableConv):
                _copy_sliced(d_blk.downsample, s_blk.downsample, full_in,
                             torch.arange(s_blk.downsample.out_channels))

    dst.fc.weight.data.copy_(src.fc.weight.data)
    dst.fc.bias.data.copy_(src.fc.bias.data)


# ---------------------------------------------------------------------------
def convert_to_lean_cifar(model, save_path=None):
    """Tao lean model tu masked model CIFAR. Tra ve (lean_model, config)."""
    model.eval()
    device = next(model.parameters()).device

    config = model.config()
    if config["family"] == "vgg":
        config["widths"] = _vgg_widths(model)
    else:
        config["block_mids"] = _resnet_mids(model)

    lean = build_cifar_model(config).to(device)
    lean.eval()
    with torch.no_grad():
        if config["family"] == "vgg":
            _surgery_vgg(model, lean)
        else:
            _surgery_resnet(model, lean)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(lean.state_dict(), save_path)
        with open(save_path.replace(".pth", ".json"), "w") as f:
            json.dump(config, f, indent=2)
    return lean, config


def demo():
    """Self-check: lean model phai cho output GIONG masked model (sai so float)."""
    from pruning import StructuredPruner

    x = torch.randn(4, 3, 32, 32)
    for name, build in (("vgg-hrank",  lambda: CifarVGG("vgg16", head="hrank")),
                        ("vgg-single", lambda: CifarVGG("vgg16", head="single")),
                        ("vgg-l1a",    lambda: CifarVGG("vgg16", prune_set="l1a")),
                        ("resnet56-A", lambda: CifarResNet(56, shortcut="A")),
                        ("resnet56-B", lambda: CifarResNet(56, shortcut="B"))):
        m = build().eval()
        StructuredPruner(m).prune(prune_ratio=0.4, verbose=False)
        with torch.no_grad():
            ref = m(x)
        lean, cfg = convert_to_lean_cifar(m)
        with torch.no_grad():
            got = lean(x)
        err = (ref - got).abs().max().item()
        assert err < 1e-4, f"{name}: surgery lech {err}"
        p0 = sum(p.numel() for p in m.parameters())
        p1 = sum(p.numel() for p in lean.parameters())
        print(f"OK  {name:<10} max|diff|={err:.2e}  "
              f"params {p0/1e6:.2f}M -> {p1/1e6:.2f}M (-{(1-p1/p0)*100:.1f}%)")


if __name__ == "__main__":
    demo()
