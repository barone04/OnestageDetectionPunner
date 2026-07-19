"""
NORTON-style CP decomposition for the standalone YOLOv1 environment.

This module keeps the detector, dataset, loss, evaluation, and finetuning code
unchanged, but replaces eligible 3x3 PrunableConv layers with CPD-based
convolutions. It is intentionally separate from channel-pruning surgery because
NORTON changes topology instead of removing channels.
"""
import ast
import copy
import math
import re
import time

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .element import PrunableConv, _make_act
from .backbone import ResNet, VGG, BasicBlock, BottleNeck
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


def cpd_weights_to_factors(pointwise_weight, vertical_weight, horizontal_weight, rank):
    """Invert CPDConv2d weights back to CP factors."""
    in_channels = pointwise_weight.size(1)
    out_channels = int(pointwise_weight.size(0) / rank)
    kernel_size = vertical_weight.size(2)

    head_factor = pointwise_weight.squeeze(-1).squeeze(-1).reshape(
        rank, out_channels, in_channels).permute(2, 0, 1)
    body_factor = vertical_weight.squeeze(1).squeeze(-1).reshape(
        rank, out_channels, kernel_size).permute(2, 0, 1)
    tail_factor = horizontal_weight.squeeze(1).squeeze(1).reshape(
        rank, out_channels, kernel_size).permute(2, 0, 1)
    return head_factor, body_factor, tail_factor


def _subspace_angles(a, b):
    a, _ = torch.linalg.qr(a)
    b, _ = torch.linalg.qr(b)
    if a.size(1) < b.size(1):
        a, b = b, a
    b = b - torch.matmul(a, torch.matmul(a.transpose(1, 0), b))
    theta = torch.asin(torch.minimum(
        torch.tensor(1.0, device=a.device), torch.norm(b)
    ))
    theta = torch.abs(theta)
    return torch.remainder(theta, torch.tensor(math.pi / 2, device=a.device))


def _factor_saliency(head_factor, body_factor, tail_factor, criterion="pabs"):
    """Match NORTON's factor-similarity pruning order."""
    num_filters = head_factor.size(2)
    saliency = torch.full((num_filters,), num_filters - 1, device=head_factor.device)

    if criterion == "vbd":
        distance_matrix = torch.zeros(num_filters, num_filters, device=head_factor.device)
        for i in range(num_filters - 1):
            for j in range(i + 1, num_filters):
                head = torch.var(head_factor[:, :, i] - head_factor[:, :, j]) / (
                    torch.var(head_factor[:, :, i]) + torch.var(head_factor[:, :, j]) + 1e-12
                )
                distance_matrix[i, j] = head
                distance_matrix[j, i] = head
        fill_value = float("inf")
        choose_min = True

    elif criterion == "csa":
        distance_matrix = torch.zeros(num_filters, num_filters, device=head_factor.device)
        for i in range(num_filters):
            for j in range(i + 1, num_filters):
                head = torch.cos(_subspace_angles(head_factor[:, :, i], head_factor[:, :, j])) ** 2
                distance_matrix[i, j] = head
                distance_matrix[j, i] = head
        fill_value = -float("inf")
        choose_min = False

    elif criterion == "pabs":
        distance_matrix = torch.zeros(num_filters, num_filters, device=head_factor.device)
        for i in range(num_filters):
            for j in range(i + 1, num_filters):
                head = _subspace_angles(head_factor[:, :, i], head_factor[:, :, j])
                body = _subspace_angles(body_factor[:, :, i], body_factor[:, :, j])
                tail = _subspace_angles(tail_factor[:, :, i], tail_factor[:, :, j])
                distance_matrix[i, j] = head + body + tail
                distance_matrix[j, i] = distance_matrix[i, j]
        fill_value = float("inf")
        choose_min = True

    else:
        raise ValueError("criterion must be one of: pabs, csa, vbd")

    distance_matrix.fill_diagonal_(fill_value)
    for i in range(num_filters - 1):
        target = torch.min(distance_matrix) if choose_min else torch.max(distance_matrix)
        rows, cols = torch.where(distance_matrix == target)
        row, col = rows[0].item(), cols[0].item()
        row_sum = torch.sum(distance_matrix[row][distance_matrix[row] != fill_value])
        col_sum = torch.sum(distance_matrix[:, col][distance_matrix[:, col] != fill_value])
        index = row if row_sum < col_sum else col
        saliency[i] = index
        distance_matrix[index, :] = fill_value
        distance_matrix[:, index] = fill_value

    return saliency


def _select_norton_filters(layer, num_keep, criterion="pabs"):
    head, body, tail = cpd_weights_to_factors(
        layer.cpd.pointwise.weight.data,
        layer.cpd.vertical.weight.data,
        layer.cpd.horizontal.weight.data,
        layer.cpd.rank,
    )
    saliency = _factor_saliency(head, body, tail, criterion)
    n = head.size(2)
    idx = torch.argsort(saliency)[n - num_keep:]
    idx, _ = idx.sort()
    return idx.detach().cpu().long()


def _select_conv_filters(layer, num_keep):
    weight = layer.conv.weight.data
    saliency = torch.norm(weight.flatten(1), dim=1)
    idx = torch.argsort(saliency)[weight.size(0) - num_keep:]
    idx, _ = idx.sort()
    return idx.detach().cpu().long()


def _keep_index(layer, prune_ratio, criterion="pabs"):
    out_channels = layer.out_channels
    # Original NORTON materializes widths with int(C * (1 - rate)).
    num_keep = max(1, int(out_channels * (1.0 - prune_ratio)))
    num_keep = min(num_keep, out_channels)
    if num_keep == out_channels:
        return torch.arange(out_channels, dtype=torch.long)
    if isinstance(layer, NortonConv):
        return _select_norton_filters(layer, num_keep, criterion)
    if isinstance(layer, PrunableConv):
        return _select_conv_filters(layer, num_keep)
    raise TypeError(f"Unsupported layer type for pruning: {type(layer)}")


def _all_out(layer):
    return torch.arange(layer.out_channels, dtype=torch.long)


def _all_in(layer):
    return torch.arange(layer.in_channels, dtype=torch.long)


def _copy_bn(dst, src, out_idx, copy_bn):
    if not copy_bn:
        return
    out_idx = out_idx.to(src.bn.weight.device)
    dst.bn.weight.data.copy_(src.bn.weight.data[out_idx])
    dst.bn.bias.data.copy_(src.bn.bias.data[out_idx])
    dst.bn.running_mean.data.copy_(src.bn.running_mean.data[out_idx])
    dst.bn.running_var.data.copy_(src.bn.running_var.data[out_idx])
    dst.bn.num_batches_tracked.data.copy_(src.bn.num_batches_tracked.data)


@torch.no_grad()
def _copy_prunable_sliced(dst, src, in_idx, out_idx, copy_bn):
    in_idx = in_idx.to(src.conv.weight.device)
    out_idx = out_idx.to(src.conv.weight.device)
    dst.conv.weight.data.copy_(src.conv.weight.data[out_idx][:, in_idx])
    if src.conv.bias is not None and dst.conv.bias is not None:
        dst.conv.bias.data.copy_(src.conv.bias.data[out_idx])
    _copy_bn(dst, src, out_idx, copy_bn)


@torch.no_grad()
def _copy_norton_sliced(dst, src, in_idx, out_idx, copy_bn):
    in_idx = in_idx.to(src.cpd.pointwise.weight.device)
    out_idx = out_idx.to(src.cpd.pointwise.weight.device)
    head, body, tail = cpd_weights_to_factors(
        src.cpd.pointwise.weight.data,
        src.cpd.vertical.weight.data,
        src.cpd.horizontal.weight.data,
        src.cpd.rank,
    )
    head = head[in_idx][:, :, out_idx]
    body = body[:, :, out_idx]
    tail = tail[:, :, out_idx]
    pointwise, vertical, horizontal = factors_to_cpd_weights(head, body, tail)
    dst.cpd.pointwise.weight.data.copy_(pointwise)
    dst.cpd.vertical.weight.data.copy_(vertical)
    dst.cpd.horizontal.weight.data.copy_(horizontal)
    if src.cpd.bias is not None and dst.cpd.bias is not None:
        dst.cpd.bias.data.copy_(src.cpd.bias.data[out_idx])
    _copy_bn(dst, src, out_idx, copy_bn)


def _copy_layer_sliced(dst, src, in_idx, out_idx, copy_bn):
    if isinstance(src, NortonConv) and isinstance(dst, NortonConv):
        _copy_norton_sliced(dst, src, in_idx, out_idx, copy_bn)
    elif isinstance(src, PrunableConv) and isinstance(dst, PrunableConv):
        _copy_prunable_sliced(dst, src, in_idx, out_idx, copy_bn)
    else:
        raise TypeError(f"Cannot copy {type(src)} -> {type(dst)}")


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
                      initialize_from_weights, module_path=""):
    replaced = 0
    for name, child in list(module.named_children()):
        child_path = f"{module_path}.{name}" if module_path else name
        if _eligible_for_norton(child):
            started = time.perf_counter()
            if initialize_from_weights:
                shape = tuple(child.conv.weight.shape)
                print(f"[CPD START] {child_path} weight={shape} "
                      f"rank={rank} device={child.conv.weight.device}", flush=True)
            new_child = prunable_to_norton(
                child, rank, n_iter_max, n_iter_singular_error,
                initialize_from_weights=initialize_from_weights,
            )
            setattr(module, name, new_child)
            replaced += 1
            if initialize_from_weights:
                elapsed = time.perf_counter() - started
                print(f"[CPD DONE]  {child_path} elapsed={elapsed:.1f}s", flush=True)
        else:
            replaced += _replace_eligible(
                child, rank, n_iter_max, n_iter_singular_error,
                initialize_from_weights, child_path,
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
    mark_norton_model(base, rank, scope)
    if "norton_prune_ratio" in config:
        base._norton_prune_ratio = float(config["norton_prune_ratio"])
    if "norton_compress_rate" in config:
        base._norton_compress_rate = [
            float(rate) for rate in config["norton_compress_rate"]
        ]
    if "norton_prune_layout" in config:
        base._norton_prune_layout = list(config["norton_prune_layout"])
    if "norton_copy_bn" in config:
        base._norton_copy_bn = bool(config["norton_copy_bn"])
    if "norton_prune_criterion" in config:
        base._norton_prune_criterion = config["norton_prune_criterion"]
    return base


def clone_as_norton_model(dense_model, rank, scope="all", n_iter_max=300,
                          n_iter_singular_error=3):
    model = copy.deepcopy(dense_model)
    return decompose_yolo_model(model, rank, scope, n_iter_max, n_iter_singular_error)


_COMPRESS_RATE_TERM = re.compile(
    r"^\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]"
    r"\s*(?:\*\s*(\d+))?$"
)


def parse_compress_rate(compress_rate):
    """Parse a NORTON compression expression or an explicit numeric list."""
    if compress_rate is None:
        return None
    if isinstance(compress_rate, (list, tuple)):
        return [float(rate) for rate in compress_rate]

    expression = str(compress_rate).strip()
    if not expression:
        raise ValueError("compress_rate cannot be empty")

    try:
        value = ast.literal_eval(expression)
    except (SyntaxError, ValueError):
        value = None
    if isinstance(value, (list, tuple)):
        return [float(rate) for rate in value]

    rates = []
    for term in expression.split("+"):
        match = _COMPRESS_RATE_TERM.fullmatch(term.strip())
        if match is None:
            raise ValueError(
                "Invalid compress_rate expression. Use an explicit list or the "
                "original NORTON form, for example '[0.]+[0.18]*12'."
            )
        rate = float(match.group(1))
        repeat = int(match.group(2) or 1)
        rates.extend([rate] * repeat)
    return rates


def get_norton_prune_layout(model, scope="all"):
    """Return the ordered YOLO width groups controlled by compress_rate."""
    if scope not in ("all", "backbone", "neck"):
        raise ValueError("scope must be one of: all, backbone, neck")

    layout = []
    if scope in ("all", "backbone"):
        if isinstance(model.backbone, ResNet):
            stages = (
                model.backbone.layer1,
                model.backbone.layer2,
                model.backbone.layer3,
                model.backbone.layer4,
            )
            for stage_index, stage in enumerate(stages, start=1):
                for block_index, block in enumerate(stage):
                    prefix = f"backbone.layer{stage_index}.{block_index}"
                    if isinstance(block, BasicBlock):
                        layout.append(f"{prefix}.conv1")
                    elif isinstance(block, BottleNeck):
                        layout.extend((f"{prefix}.conv1", f"{prefix}.conv2"))
                    else:
                        raise TypeError(f"Unsupported ResNet block: {type(block)}")
        elif isinstance(model.backbone, VGG):
            for layer_index, layer in enumerate(model.backbone.features):
                if isinstance(layer, (PrunableConv, NortonConv)):
                    layout.append(f"backbone.features.{layer_index}")
        else:
            raise TypeError(f"Unsupported backbone: {type(model.backbone)}")

    if scope in ("all", "neck"):
        for layer_index, layer in enumerate(model.neck.convs):
            if isinstance(layer, (PrunableConv, NortonConv)):
                layout.append(f"neck.convs.{layer_index}")
    return layout


def normalize_norton_compress_rate(model, prune_ratio=0.0,
                                   compress_rate=None, scope="all"):
    """Resolve uniform or per-layer pruning into the ordered rate vector."""
    layout = get_norton_prune_layout(model, scope)
    parsed = parse_compress_rate(compress_rate)

    if parsed is None:
        rate = float(prune_ratio or 0.0)
        rates = [rate] * len(layout)
    else:
        if prune_ratio is not None and float(prune_ratio) != 0.0:
            raise ValueError("Use either prune_ratio or compress_rate, not both")
        rates = parsed

    if len(rates) != len(layout):
        raise ValueError(
            f"scope={scope} requires {len(layout)} compression rates, "
            f"got {len(rates)}. Ordered layout: {layout}"
        )
    rates = [float(rate) for rate in rates]
    if any(not math.isfinite(rate) or rate < 0.0 or rate >= 1.0 for rate in rates):
        raise ValueError("Every compression rate must be finite and in [0, 1)")
    return rates, layout


def _resnet_plan(backbone, compress_rate, criterion, enabled):
    block_mids = []
    selectors = []
    rate_index = 0
    for stage in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4):
        for block in stage:
            if isinstance(block, BasicBlock):
                rate = compress_rate[rate_index] if enabled else 0.0
                conv1_idx = _keep_index(block.conv1, rate, criterion) if enabled else _all_out(block.conv1)
                rate_index += int(enabled)
                block_mids.append(int(conv1_idx.numel()))
                selectors.append({"conv1": conv1_idx})
            elif isinstance(block, BottleNeck):
                rate1 = compress_rate[rate_index] if enabled else 0.0
                rate2 = compress_rate[rate_index + 1] if enabled else 0.0
                conv1_idx = _keep_index(block.conv1, rate1, criterion) if enabled else _all_out(block.conv1)
                conv2_idx = _keep_index(block.conv2, rate2, criterion) if enabled else _all_out(block.conv2)
                rate_index += 2 * int(enabled)
                block_mids.append([int(conv1_idx.numel()), int(conv2_idx.numel())])
                selectors.append({"conv1": conv1_idx, "conv2": conv2_idx})
            else:
                raise TypeError(f"Unsupported ResNet block: {type(block)}")
    return block_mids, selectors


def _vgg_plan(backbone, compress_rate, criterion, enabled):
    widths = []
    selectors = []
    rate_index = 0
    for layer in backbone.features:
        if isinstance(layer, (PrunableConv, NortonConv)):
            rate = compress_rate[rate_index] if enabled else 0.0
            idx = _keep_index(layer, rate, criterion) if enabled else _all_out(layer)
            rate_index += int(enabled)
            widths.append(int(idx.numel()))
            selectors.append(idx)
    return widths, selectors


def _neck_plan(neck, compress_rate, criterion, enabled):
    widths = []
    selectors = []
    rate_index = 0
    for layer in neck.convs:
        rate = compress_rate[rate_index] if enabled else 0.0
        idx = _keep_index(layer, rate, criterion) if enabled else _all_out(layer)
        rate_index += int(enabled)
        widths.append(int(idx.numel()))
        selectors.append(idx)
    return widths, selectors


def _make_prune_plan(model, prune_ratio, compress_rate, criterion, scope,
                     copy_bn):
    prune_backbone = scope in ("all", "backbone")
    prune_neck = scope in ("all", "neck")
    rates, layout = normalize_norton_compress_rate(
        model, prune_ratio=prune_ratio, compress_rate=compress_rate, scope=scope
    )
    backbone_rate_count = sum(name.startswith("backbone.") for name in layout)
    backbone_rates = rates[:backbone_rate_count]
    neck_rates = rates[backbone_rate_count:]
    plan = {
        "scope": scope,
        "criterion": criterion,
        "prune_ratio": float(prune_ratio) if compress_rate is None else None,
        "compress_rate": rates,
        "prune_layout": layout,
        "copy_bn": bool(copy_bn),
        "backbone_type": "resnet" if isinstance(model.backbone, ResNet) else "vgg",
    }

    if isinstance(model.backbone, ResNet):
        bb_cfg, bb_selectors = _resnet_plan(
            model.backbone, backbone_rates, criterion, prune_backbone
        )
    elif isinstance(model.backbone, VGG):
        bb_cfg, bb_selectors = _vgg_plan(
            model.backbone, backbone_rates, criterion, prune_backbone
        )
    else:
        raise TypeError(f"Unsupported backbone: {type(model.backbone)}")

    neck_cfg, neck_selectors = _neck_plan(
        model.neck, neck_rates, criterion, prune_neck
    )

    plan["backbone_cfg"] = bb_cfg if prune_backbone else model._backbone_cfg
    plan["backbone_selectors"] = bb_selectors
    plan["neck_widths"] = neck_cfg if prune_neck else model._neck_widths
    plan["neck_selectors"] = neck_selectors
    return plan


def _norton_config_from_plan(model, plan):
    config = {
        "variant": "norton",
        "backbone": model.backbone_name,
        "input_size": model.input_size,
        "num_classes": model.num_classes,
        "neck_out": model._neck_out,
        "backbone_cfg": plan["backbone_cfg"],
        "neck_widths": plan["neck_widths"],
        "norton_rank": int(model._norton_rank),
        "norton_scope": model._norton_scope,
        "norton_compress_rate": plan["compress_rate"],
        "norton_prune_layout": plan["prune_layout"],
        "norton_copy_bn": plan["copy_bn"],
        "norton_prune_criterion": plan["criterion"],
    }
    if plan["prune_ratio"] is not None:
        config["norton_prune_ratio"] = float(plan["prune_ratio"])
    return config


def _copy_resnet_backbone(src, dst, selectors, copy_bn):
    _copy_layer_sliced(
        dst.conv1, src.conv1, _all_in(src.conv1), _all_out(src.conv1), copy_bn
    )
    ptr = 0
    for src_stage, dst_stage in zip(
        (src.layer1, src.layer2, src.layer3, src.layer4),
        (dst.layer1, dst.layer2, dst.layer3, dst.layer4),
    ):
        for src_block, dst_block in zip(src_stage, dst_stage):
            sel = selectors[ptr]
            ptr += 1
            if isinstance(src_block, BasicBlock):
                mid_idx = sel["conv1"]
                _copy_layer_sliced(dst_block.conv1, src_block.conv1,
                                   _all_in(src_block.conv1), mid_idx, copy_bn)
                _copy_layer_sliced(dst_block.conv2, src_block.conv2,
                                   mid_idx, _all_out(src_block.conv2), copy_bn)
            else:
                mid1_idx = sel["conv1"]
                mid2_idx = sel["conv2"]
                _copy_layer_sliced(dst_block.conv1, src_block.conv1,
                                   _all_in(src_block.conv1), mid1_idx, copy_bn)
                _copy_layer_sliced(dst_block.conv2, src_block.conv2,
                                   mid1_idx, mid2_idx, copy_bn)
                _copy_layer_sliced(dst_block.conv3, src_block.conv3,
                                   mid2_idx, _all_out(src_block.conv3), copy_bn)

            if isinstance(src_block.downsample, PrunableConv):
                _copy_layer_sliced(dst_block.downsample, src_block.downsample,
                                   _all_in(src_block.downsample),
                                   _all_out(src_block.downsample), copy_bn)

    last_block = src.layer4[-1]
    out_ch = last_block.conv2.out_channels if isinstance(last_block, BasicBlock) else last_block.conv3.out_channels
    return torch.arange(out_ch, dtype=torch.long)


def _copy_vgg_backbone(src, dst, selectors, copy_bn):
    prev = torch.arange(3, dtype=torch.long)
    src_convs = [m for m in src.features if isinstance(m, (PrunableConv, NortonConv))]
    dst_convs = [m for m in dst.features if isinstance(m, (PrunableConv, NortonConv))]
    for src_layer, dst_layer, out_idx in zip(src_convs, dst_convs, selectors):
        _copy_layer_sliced(dst_layer, src_layer, prev, out_idx, copy_bn)
        prev = out_idx
    return prev


def _copy_neck(src, dst, selectors, input_idx, copy_bn):
    prev = input_idx
    for src_layer, dst_layer, out_idx in zip(src.convs, dst.convs, selectors):
        _copy_layer_sliced(dst_layer, src_layer, prev, out_idx, copy_bn)
        prev = out_idx
    return prev


@torch.no_grad()
def _copy_head(src, dst, in_idx):
    in_idx = in_idx.to(src.detect.weight.device)
    dst.detect.weight.data.copy_(src.detect.weight.data[:, in_idx])
    if src.detect.bias is not None:
        dst.detect.bias.data.copy_(src.detect.bias.data)


def prune_norton_model(model, prune_ratio=0.0, criterion="pabs", scope="all",
                       compress_rate=None, copy_bn=False):
    """
    Materialize a smaller NORTON model by pruning channel/filter selections.

    The selection follows the original NORTON idea for CPD layers: score filters
    through CP factors, then rebuild a dense smaller topology and slice weights.
    """
    if getattr(model, "_variant", None) != "norton":
        raise ValueError("prune_norton_model expects a decomposed NORTON model")
    if compress_rate is None and not (0.0 <= prune_ratio < 1.0):
        raise ValueError("prune_ratio must be in [0, 1)")
    if scope not in ("all", "backbone", "neck"):
        raise ValueError("scope must be one of: all, backbone, neck")

    device = next(model.parameters()).device
    plan = _make_prune_plan(
        model, prune_ratio, compress_rate, criterion, scope, copy_bn
    )
    config = _norton_config_from_plan(model, plan)
    pruned = build_norton_model_from_config(config).to(device)

    with torch.no_grad():
        if isinstance(model.backbone, ResNet):
            bb_out_idx = _copy_resnet_backbone(
                model.backbone, pruned.backbone, plan["backbone_selectors"],
                copy_bn,
            )
        else:
            bb_out_idx = _copy_vgg_backbone(
                model.backbone, pruned.backbone, plan["backbone_selectors"],
                copy_bn,
            )
        neck_out_idx = _copy_neck(
            model.neck, pruned.neck, plan["neck_selectors"], bb_out_idx,
            copy_bn,
        )
        _copy_head(model.head, pruned.head, neck_out_idx)

    if compress_rate is None:
        pruned._norton_prune_ratio = float(prune_ratio)
    pruned._norton_compress_rate = list(plan["compress_rate"])
    pruned._norton_prune_layout = list(plan["prune_layout"])
    pruned._norton_copy_bn = bool(copy_bn)
    pruned._norton_prune_criterion = criterion
    return pruned, config
