"""
Khoi xay co ban (standalone) cho one-stage detector co the cat tia (prune).

- PrunableConv = Conv2d + BN + Activation, kem 2 mask:
    * u_mask (Unstructured, 4D)  -> Song Han magnitude pruning
    * s_mask (Structured, 1D)    -> Filter/channel pruning
  Mask duoc ap (mul_) vao conv.weight NGAY trong forward => zero giu vung khi finetune.
- MaskProxy: dieu huong pruner toi dung u_mask hoac s_mask cua layer.

Khong ke thua / khong import tu PrunedFishNet hay YOLOv1 (chi tham khao y tuong).
"""
import torch
import torch.nn as nn


def _make_act(act: str):
    if act in (None, "identity"):
        return nn.Identity()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "leaky":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unknown act: {act}")


class UnstructuredMask(nn.Module):
    """Mask 4D (theo tung trong so) cho Song Han pruning."""

    def __init__(self, weight_shape):
        super().__init__()
        self.register_buffer("mask", torch.ones(weight_shape))

    def update(self, new_mask):
        self.mask.data.copy_(new_mask)

    def apply(self, conv):
        conv.weight.data.mul_(self.mask)


class StructuredMask(nn.Module):
    """Mask 1D (theo out-channel/filter) cho Filter pruning."""

    def __init__(self, out_channels):
        super().__init__()
        self.register_buffer("mask", torch.ones(out_channels))

    def update(self, new_mask):
        self.mask.data.copy_(new_mask)

    def apply(self, conv):
        conv.weight.data.mul_(self.mask.view(-1, 1, 1, 1))


class MaskProxy:
    """
    Lop trung gian cho pruner. Pruner thao tac qua proxy; proxy lazy-init va dieu huong
    toi dung u_mask / s_mask cua layer. API: .conv, .mask_handler(.update/.apply).
    """

    def __init__(self, layer, mask_type):
        self.layer = layer
        self.mask_type = mask_type

    @property
    def mask_handler(self):
        if self.mask_type == "unstructured":
            if self.layer.u_mask is None:
                self.layer.u_mask = UnstructuredMask(self.layer.conv.weight.shape).to(
                    self.layer.conv.weight.device
                )
            return self.layer.u_mask
        elif self.mask_type == "structured":
            if self.layer.s_mask is None:
                self.layer.s_mask = StructuredMask(self.layer.conv.out_channels).to(
                    self.layer.conv.weight.device
                )
            return self.layer.s_mask
        return None

    @property
    def conv(self):
        return self.layer.conv

    def __getattr__(self, name):
        return getattr(self.layer, name)


class PrunableConv(nn.Module):
    """Conv2d + BN + Act, ho tro dual masking (Unstructured + Structured)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, bias=False, act="leaky"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act_name = act if act is not None else "identity"
        self.act = _make_act(act)

        self.u_mask = None   # Unstructured
        self.s_mask = None   # Structured

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def weight(self):
        return self.conv.weight

    def forward(self, x):
        if self.u_mask is not None:
            self.u_mask.apply(self.conv)
        if self.s_mask is not None:
            self.s_mask.apply(self.conv)
        return self.act(self.bn(self.conv(x)))

    def get_prunable_layers(self, pruning_type="unstructured"):
        return [MaskProxy(self, pruning_type)]
