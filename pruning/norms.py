"""
Chuan mixed-norm L1-inf-inf cho filter pruning (structured).

l1inftyinfty(filter (C_in,H,W)) = sum_{C_in} max_{H,W} |w|
  = max over (H,W) cho moi kenh dau vao, roi sum theo kenh dau vao.
Cac ham nhan ca tensor 3D (1 filter) lan 4D (N filter) — vectorized.
"""
import torch


def l1inftyinfty(weight):
    """
    weight: (..., C_in, H, W) -> tra (...,) scalar-norm cho moi filter.
    """
    return weight.abs().amax(dim=(-2, -1)).sum(dim=-1)


def l1inftyinfty_distance(f1, f2):
    return l1inftyinfty((f1 - f2).abs())


def get_weight(layer):
    """Lay weight tu PrunableConv (co .conv) hoac Conv2d thuong."""
    if hasattr(layer, "conv"):
        return layer.conv.weight
    if hasattr(layer, "weight"):
        return layer.weight
    raise ValueError(f"Cannot extract weight from {type(layer)}")
