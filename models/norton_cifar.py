"""NORTON ResNet-56 implementation for CIFAR-10 validation.

This module mirrors the ResNet-56 topology and 30-value compression mapping in
the original NORTON repository. It is deliberately separate from the YOLO
models because this path validates the algorithm on its native benchmark.
"""
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norton import (
    CPDConv2d,
    _factor_saliency,
    conv_weights_to_factors,
    cpd_weights_to_factors,
    factors_to_cpd_weights,
)


NUM_BLOCKS = 9
NUM_COMPRESS_RATES = 30


def normalize_compress_rate(compress_rate=None):
    rates = [0.0] * NUM_COMPRESS_RATES if compress_rate is None else list(compress_rate)
    if len(rates) != NUM_COMPRESS_RATES:
        raise ValueError(
            f"ResNet-56 requires {NUM_COMPRESS_RATES} compression rates, got {len(rates)}"
        )
    rates = [float(rate) for rate in rates]
    if any(rate < 0.0 or rate >= 1.0 for rate in rates):
        raise ValueError("Every compression rate must be in [0, 1)")
    return rates


def adapt_channels(compress_rate):
    """Match models/cifar10/resnet.py::adapt_channel from original NORTON."""
    rates = normalize_compress_rate(compress_rate)
    stage_out_channels = [16] + [16] * 9 + [32] * 9 + [64] * 9
    stage_out_rates = [rates[0]] + [rates[1]] * 9 + [rates[2]] * 9 + [0.0] * 9
    mid_rates = rates[3:]

    overall_channels = [
        int(channels * (1.0 - rate))
        for channels, rate in zip(stage_out_channels, stage_out_rates)
    ]
    mid_channels = [
        int(stage_out_channels[index + 1] * (1.0 - mid_rates[index]))
        for index in range(27)
    ]
    if min(overall_channels + mid_channels) <= 0:
        raise ValueError("Compression rates produced a zero-width layer")
    return overall_channels, mid_channels


def conv3x3(in_channels, out_channels, rank=0, stride=1):
    if rank <= 0:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
    return CPDConv2d(
        in_channels, out_channels, rank=rank, kernel_size=3,
        stride=stride, padding=1, bias=False,
    )


class OptionAShortcut(nn.Module):
    """CIFAR ResNet Option-A shortcut used by the original NORTON model."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        if self.stride != 1:
            x = x[:, :, ::2, ::2]
        delta = self.out_channels - self.in_channels
        left = delta // 2
        right = delta - left
        return F.pad(x, (0, 0, 0, 0, left, right), "constant", 0)


class CifarBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, mid_channels, in_channels, out_channels, rank=0, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, mid_channels, rank=rank, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_channels, out_channels, rank=rank)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else OptionAShortcut(in_channels, out_channels, stride)
        )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + self.shortcut(x))


class NortonCifarResNet56(nn.Module):
    def __init__(self, compress_rate=None, rank=0, num_classes=10):
        super().__init__()
        self.compress_rate = normalize_compress_rate(compress_rate)
        self.rank = int(rank)
        self.num_classes = int(num_classes)
        overall, mid = adapt_channels(self.compress_rate)
        self.overall_channels = overall
        self.mid_channels = mid
        self._channel_ptr = 0

        self.conv1 = conv3x3(3, overall[0], rank=self.rank)
        self.bn1 = nn.BatchNorm2d(overall[0])
        self.relu = nn.ReLU(inplace=True)
        self._channel_ptr = 1
        self.layer1 = self._make_stage(stride=1)
        self.layer2 = self._make_stage(stride=2)
        self.layer3 = self._make_stage(stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, self.num_classes)

    def _make_stage(self, stride):
        blocks = []
        for block_index in range(NUM_BLOCKS):
            ptr = self._channel_ptr
            blocks.append(CifarBasicBlock(
                self.mid_channels[ptr - 1],
                self.overall_channels[ptr - 1],
                self.overall_channels[ptr],
                rank=self.rank,
                stride=stride if block_index == 0 else 1,
            ))
            self._channel_ptr += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

    def config(self):
        return {
            "arch": "resnet56",
            "dataset": "cifar10",
            "rank": self.rank,
            "compress_rate": self.compress_rate,
            "num_classes": self.num_classes,
        }


def build_cifar_resnet56(config=None, *, compress_rate=None, rank=0, num_classes=10):
    if config is not None:
        compress_rate = config.get("compress_rate", compress_rate)
        rank = config.get("rank", rank)
        num_classes = config.get("num_classes", num_classes)
    return NortonCifarResNet56(compress_rate, rank, num_classes)


@torch.no_grad()
def _conv_to_cpd(conv, rank, n_iter_max, n_iter_singular_error):
    cpd = CPDConv2d(
        conv.in_channels, conv.out_channels, rank,
        kernel_size=conv.kernel_size[0], stride=conv.stride[0],
        padding=conv.padding[0], bias=conv.bias is not None,
        device=conv.weight.device,
    ).to(conv.weight.device)
    head, body, tail = conv_weights_to_factors(
        conv.weight.data, rank, n_iter_max, n_iter_singular_error
    )
    pointwise, vertical, horizontal = factors_to_cpd_weights(head, body, tail)
    cpd.pointwise.weight.data.copy_(pointwise)
    cpd.vertical.weight.data.copy_(vertical)
    cpd.horizontal.weight.data.copy_(horizontal)
    if conv.bias is not None:
        cpd.bias.data.copy_(conv.bias.data)
    return cpd


def decompose_cifar_resnet56(model, rank, n_iter_max=300, n_iter_singular_error=3):
    """Replace every 3x3 Conv2d with an initialized CPDConv2d exactly once."""
    model = copy.deepcopy(model)
    replaced = 0

    def replace(module, module_path=""):
        nonlocal replaced
        for name, child in list(module.named_children()):
            child_path = f"{module_path}.{name}" if module_path else name
            if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
                started = time.perf_counter()
                print(f"[CPD START] {child_path} weight={tuple(child.weight.shape)} "
                      f"rank={rank} device={child.weight.device}", flush=True)
                setattr(module, name, _conv_to_cpd(
                    child, rank, n_iter_max, n_iter_singular_error
                ))
                replaced += 1
                elapsed = time.perf_counter() - started
                print(f"[CPD DONE]  {child_path} elapsed={elapsed:.1f}s", flush=True)
            else:
                replace(child, child_path)

    replace(model)
    model.rank = int(rank)
    return model, replaced


def _all_indices(size):
    return torch.arange(size, dtype=torch.long)


def _copy_bn(dst, src, indices):
    indices = indices.to(src.weight.device)
    dst.weight.data.copy_(src.weight.data[indices])
    dst.bias.data.copy_(src.bias.data[indices])
    dst.running_mean.data.copy_(src.running_mean.data[indices])
    dst.running_var.data.copy_(src.running_var.data[indices])
    dst.num_batches_tracked.data.copy_(src.num_batches_tracked.data)


@torch.no_grad()
def _materialize_cpd(dst, src, input_indices, output_indices):
    input_indices = input_indices.to(src.pointwise.weight.device)
    output_indices = output_indices.to(src.pointwise.weight.device)
    head, body, tail = cpd_weights_to_factors(
        src.pointwise.weight.data,
        src.vertical.weight.data,
        src.horizontal.weight.data,
        src.rank,
    )
    head = head[input_indices][:, :, output_indices]
    body = body[:, :, output_indices]
    tail = tail[:, :, output_indices]
    pointwise, vertical, horizontal = factors_to_cpd_weights(head, body, tail)
    dst.pointwise.weight.data.copy_(pointwise)
    dst.vertical.weight.data.copy_(vertical)
    dst.horizontal.weight.data.copy_(horizontal)
    if src.bias is not None and dst.bias is not None:
        dst.bias.data.copy_(src.bias.data[output_indices])


def _select_outputs(src, input_indices, num_keep, criterion):
    head, body, tail = cpd_weights_to_factors(
        src.pointwise.weight.data,
        src.vertical.weight.data,
        src.horizontal.weight.data,
        src.rank,
    )
    head = head[input_indices.to(head.device)]
    saliency = _factor_saliency(head, body, tail, criterion)
    selected = torch.argsort(saliency)[head.size(2) - num_keep:]
    selected, _ = selected.sort()
    return selected.detach().cpu().long()


@torch.no_grad()
def prune_cifar_resnet56(model, compress_rate, criterion="pabs", copy_bn=False):
    """Apply original NORTON one-shot factor pruning to decomposed ResNet-56.

    ``copy_bn=False`` mirrors the released NORTON ResNet pruning code, where the
    compressed model starts with freshly initialized BN parameters/statistics.
    """
    rates = normalize_compress_rate(compress_rate)
    if rates[0] != 0.0:
        raise ValueError("Original ResNet-56 pruning keeps the stem: compress_rate[0] must be 0")
    if model.rank <= 0 or not isinstance(model.conv1, CPDConv2d):
        raise ValueError("Expected a decomposed ResNet-56 model")

    device = next(model.parameters()).device
    pruned = build_cifar_resnet56(
        compress_rate=rates, rank=model.rank, num_classes=model.num_classes
    ).to(device)

    stem_in = _all_indices(model.conv1.in_channels)
    stem_out = _all_indices(model.conv1.out_channels)
    _materialize_cpd(pruned.conv1, model.conv1, stem_in, stem_out)
    if copy_bn:
        _copy_bn(pruned.bn1, model.bn1, stem_out)

    previous_output = None
    for src_stage, dst_stage in zip(
        (model.layer1, model.layer2, model.layer3),
        (pruned.layer1, pruned.layer2, pruned.layer3),
    ):
        for src_block, dst_block in zip(src_stage, dst_stage):
            for src_conv, dst_conv, src_bn, dst_bn in (
                (src_block.conv1, dst_block.conv1, src_block.bn1, dst_block.bn1),
                (src_block.conv2, dst_block.conv2, src_block.bn2, dst_block.bn2),
            ):
                input_indices = (
                    previous_output
                    if previous_output is not None
                    else _all_indices(src_conv.in_channels)
                )
                if dst_conv.out_channels < src_conv.out_channels:
                    output_indices = _select_outputs(
                        src_conv, input_indices, dst_conv.out_channels, criterion
                    )
                    previous_output = output_indices
                else:
                    output_indices = _all_indices(src_conv.out_channels)
                    previous_output = None

                _materialize_cpd(dst_conv, src_conv, input_indices, output_indices)
                if copy_bn:
                    _copy_bn(dst_bn, src_bn, output_indices)

    pruned.fc.weight.data.copy_(model.fc.weight.data)
    pruned.fc.bias.data.copy_(model.fc.bias.data)
    return pruned
