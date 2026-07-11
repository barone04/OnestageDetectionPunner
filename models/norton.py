"""
NORTON-style CP decomposition for the standalone YOLOv1 environment.

This module keeps the detector, dataset, loss, evaluation, and finetuning code
unchanged, but replaces eligible 3x3 PrunableConv layers with CPD-based
convolutions. It is intentionally separate from channel-pruning surgery because
NORTON changes topology instead of removing channels.
"""
import copy

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .element import PrunableConv, _make_act
from .yolo import YoloModel


class CPDConv2d(nn.Module):
    """CPD approximation of a regular square Conv2d layer."""

    def __init__(self, in_channels, out_channels, rank, kernel_size=3,
                 stride=1, padding=1, bias=False, device=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        channels = rank * out_channels

        self.pointwise = nn.Conv2d(in_channels, channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        self.vertical = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                                  stride=(stride, 1), padding=(padding, 0),
                                  groups=channels, bias=False)
        self.horizontal = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                                    stride=(1, stride), padding=(0, padding),
                                    groups=channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, device=device))
        else:
            self.bias = None

    def forward(self, x):
        out = self.pointwise(x)
        out = self.vertical(out)
        out = self.horizontal(out)
        out = out.view(out.shape[0], self.rank, self.out_channels, *out.shape[2:])
        out = torch.sum(out, dim=1)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class NortonConv(nn.Module):
    """CPDConv2d + BN + activation, matching the PrunableConv interface enough for YOLO."""

    def __init__(self, in_channels, out_channels, rank, kernel_size=3,
                 stride=1, padding=1, bias=False, act="leaky", device=None):
        super().__init__()
        self.cpd = CPDConv2d(in_channels, out_channels, rank, kernel_size,
                             stride, padding, bias=bias, device=device)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act_name = act if act is not None else "identity"
        self.act = _make_act(act)

    @property
    def out_channels(self):
        return self.cpd.out_channels

    @property
    def in_channels(self):
        return self.cpd.in_channels

    def forward(self, x):
        return self.act(self.bn(self.cpd(x)))


def _require_tensorly():
    try:
        import tensorly as tl
        from tensorly.decomposition import parafac
    except ImportError as exc:
        raise ImportError(
            "NORTON decomposition requires tensorly. Install requirements.txt "
            "or run: pip install tensorly"
        ) from exc
    tl.set_backend("pytorch")
    return parafac


def conv_weights_to_factors(weights, rank, n_iter_max=300, n_iter_singular_error=3):
    """
    Decompose Conv2d weights into CP factors per output filter.

    Returns factors with shapes:
      head: (in_channels, rank, out_channels)
      body: (kernel_size, rank, out_channels)
      tail: (kernel_size, rank, out_channels)
    """
    parafac = _require_tensorly()
    kernel_size = weights.size(2)
    in_channels = weights.size(1)
    out_channels = weights.size(0)
    device = weights.device

    head_factor = torch.zeros((in_channels, rank, out_channels), device=device)
    body_factor = torch.zeros((kernel_size, rank, out_channels), device=device)
    tail_factor = torch.zeros((kernel_size, rank, out_channels), device=device)

    for i in tqdm(range(out_channels), desc="CPD filters"):
        weight = weights[i]
        if not torch.any(weight):
            continue
        success = False
        count = 0
        while not success and count < n_iter_singular_error:
            try:
                _, factors = parafac(weight, rank=rank, n_iter_max=n_iter_max,
                                     init="random")
                head_factor[:, :, i], body_factor[:, :, i], tail_factor[:, :, i] = factors
                success = True
            except torch._C._LinAlgError:
                count += 1
        if not success:
            raise RuntimeError(f"CPD failed for output filter {i}")

    for name, factor in (("head", head_factor), ("body", body_factor), ("tail", tail_factor)):
        if torch.isnan(factor).any():
            raise RuntimeError(f"{name}_factor from CPD contains NaN")
    return head_factor, body_factor, tail_factor


def _transform_factor(factor):
    rank = factor.size(1)
    out_channels = factor.size(2)
    x = factor.size(0)
    return factor.reshape(x, rank * out_channels).permute(1, 0)


def factors_to_cpd_weights(head_factor, body_factor, tail_factor):
    pointwise = _transform_factor(head_factor).unsqueeze(-1).unsqueeze(-1)
    vertical = _transform_factor(body_factor).unsqueeze(1).unsqueeze(-1)
    horizontal = _transform_factor(tail_factor).unsqueeze(1).unsqueeze(2)
    return pointwise, vertical, horizontal


@torch.no_grad()
def prunable_to_norton(layer, rank, n_iter_max=300, n_iter_singular_error=3,
                       initialize_from_weights=True):
    conv = layer.conv
    device = conv.weight.device
    k_h, k_w = conv.kernel_size
    if k_h != k_w:
        raise ValueError(f"NORTON expects square kernels, got {conv.kernel_size}")

    out = NortonConv(conv.in_channels, conv.out_channels, rank, kernel_size=k_h,
                     stride=conv.stride[0], padding=conv.padding[0],
                     bias=(conv.bias is not None), act=layer.act_name,
                     device=device).to(device)
    if initialize_from_weights:
        head, body, tail = conv_weights_to_factors(
            conv.weight.data, rank, n_iter_max, n_iter_singular_error
        )
        pw, vh, hh = factors_to_cpd_weights(head, body, tail)
        out.cpd.pointwise.weight.data.copy_(pw)
        out.cpd.vertical.weight.data.copy_(vh)
        out.cpd.horizontal.weight.data.copy_(hh)
        if conv.bias is not None and out.cpd.bias is not None:
            out.cpd.bias.data.copy_(conv.bias.data)

    out.bn.weight.data.copy_(layer.bn.weight.data)
    out.bn.bias.data.copy_(layer.bn.bias.data)
    out.bn.running_mean.data.copy_(layer.bn.running_mean.data)
    out.bn.running_var.data.copy_(layer.bn.running_var.data)
    out.bn.num_batches_tracked.data.copy_(layer.bn.num_batches_tracked.data)
    return out


def _eligible_for_norton(layer):
    if not isinstance(layer, PrunableConv):
        return False
    return tuple(layer.conv.kernel_size) == (3, 3)


def _replace_eligible(module, rank, n_iter_max, n_iter_singular_error,
                      initialize_from_weights):
    replaced = 0
    for name, child in list(module.named_children()):
        if _eligible_for_norton(child):
            new_child = prunable_to_norton(
                child, rank, n_iter_max, n_iter_singular_error,
                initialize_from_weights=initialize_from_weights,
            )
            setattr(module, name, new_child)
            replaced += 1
        else:
            replaced += _replace_eligible(
                child, rank, n_iter_max, n_iter_singular_error,
                initialize_from_weights,
            )
    return replaced


def mark_norton_model(model, rank, scope):
    model._variant = "norton"
    model._norton_rank = int(rank)
    model._norton_scope = scope
    return model


def decompose_yolo_model(model, rank, scope="all", n_iter_max=300,
                         n_iter_singular_error=3):
    """Replace eligible 3x3 PrunableConv layers with initialized NortonConv layers."""
    if scope not in ("all", "backbone", "neck"):
        raise ValueError("scope must be one of: all, backbone, neck")
    replaced = 0
    if scope in ("all", "backbone"):
        replaced += _replace_eligible(model.backbone, rank, n_iter_max,
                                      n_iter_singular_error, True)
    if scope in ("all", "neck"):
        replaced += _replace_eligible(model.neck, rank, n_iter_max,
                                      n_iter_singular_error, True)
    mark_norton_model(model, rank, scope)
    return model, replaced


def build_norton_model_from_config(config):
    """Instantiate a NORTON topology from config, ready for state_dict loading."""
    rank = int(config.get("norton_rank", config.get("rank", 0)))
    scope = config.get("norton_scope", config.get("scope", "all"))
    if rank <= 0:
        raise ValueError("NORTON config requires a positive norton_rank")

    base = YoloModel(
        input_size=config.get("input_size", 448),
        backbone=config.get("backbone", "resnet18"),
        num_classes=config.get("num_classes", 1),
        neck_out=config.get("neck_out", 512),
        backbone_cfg=config.get("backbone_cfg", None),
        neck_widths=config.get("neck_widths", None),
    )
    if scope in ("all", "backbone"):
        _replace_eligible(base.backbone, rank, 0, 0, initialize_from_weights=False)
    if scope in ("all", "neck"):
        _replace_eligible(base.neck, rank, 0, 0, initialize_from_weights=False)
    return mark_norton_model(base, rank, scope)


def clone_as_norton_model(dense_model, rank, scope="all", n_iter_max=300,
                          n_iter_singular_error=3):
    model = copy.deepcopy(dense_model)
    return decompose_yolo_model(model, rank, scope, n_iter_max, n_iter_singular_error)
