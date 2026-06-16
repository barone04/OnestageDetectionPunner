"""
Model surgery — materialize masked model thanh LEAN model (dense nho that su).

Y tuong: structured mask (s_mask) danh dau filter bi cat. Surgery:
  1. Trich config so kenh con song (width) cho moi prunable conv.
  2. Dung lai YoloModel voi config nho.
  3. Copy weight, slice theo index kenh con song:
       - output slice theo s_mask cua chinh conv,
       - input  slice theo s_mask cua conv PRODUCER (theo chuoi phu thuoc).

Chuoi phu thuoc YOLOv1:
  ResNet backbone: stem & output block GIU NGUYEN (residual). Chi mid-channel block giam.
                   => backbone output = full.
  VGG  backbone : chuoi tuyen tinh; conv_k.in = conv_{k-1}.out (kept).
                   => backbone output = last conv (kept).
  Neck (5 conv) : chuoi tuyen tinh; conv0.in = backbone output; ...; conv4.out -> head.in
  Head (1x1)    : output GIU NGUYEN (5+C); input = neck conv4 (kept).
"""
import os
import json
import copy

import torch

from models.yolo import YoloModel
from models.element import PrunableConv
from models.backbone import ResNet, VGG, BasicBlock, BottleNeck


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _kept_idx(pconv: PrunableConv):
    """Index out-channel con song theo s_mask (None -> giu het). Toi thieu giu 1 kenh."""
    if pconv.s_mask is None:
        return torch.arange(pconv.out_channels)
    idx = torch.nonzero(pconv.s_mask.mask, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        idx = torch.tensor([0], dtype=torch.long)
    return idx.long()


def _copy_sliced(dst: PrunableConv, src: PrunableConv, in_idx, out_idx):
    in_idx = in_idx.to(src.conv.weight.device)
    out_idx = out_idx.to(src.conv.weight.device)
    w = src.conv.weight.data[out_idx][:, in_idx]
    dst.conv.weight.data.copy_(w)
    if src.conv.bias is not None and dst.conv.bias is not None:
        dst.conv.bias.data.copy_(src.conv.bias.data[out_idx])
    dst.bn.weight.data.copy_(src.bn.weight.data[out_idx])
    dst.bn.bias.data.copy_(src.bn.bias.data[out_idx])
    dst.bn.running_mean.data.copy_(src.bn.running_mean.data[out_idx])
    dst.bn.running_var.data.copy_(src.bn.running_var.data[out_idx])
    dst.bn.num_batches_tracked.data.copy_(src.bn.num_batches_tracked.data)


# ---------------------------------------------------------------------------
# config extraction (widths con song)
# ---------------------------------------------------------------------------
def _extract_resnet_cfg(backbone: ResNet):
    block_mids = []
    for stage in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4):
        for blk in stage:
            if isinstance(blk, BasicBlock):
                block_mids.append(int(_kept_idx(blk.conv1).numel()))
            else:  # BottleNeck
                block_mids.append([int(_kept_idx(blk.conv1).numel()),
                                   int(_kept_idx(blk.conv2).numel())])
    return block_mids


def _extract_vgg_cfg(backbone: VGG):
    return [int(_kept_idx(m).numel()) for m in backbone.features if isinstance(m, PrunableConv)]


def _extract_neck_cfg(neck):
    return [int(_kept_idx(m).numel()) for m in neck.convs if isinstance(m, PrunableConv)]


# ---------------------------------------------------------------------------
# per-component weight copy
# ---------------------------------------------------------------------------
def _surgery_backbone(src_bb, dst_bb):
    """Copy weight backbone, tra ve index kenh OUTPUT con song (cho neck input)."""
    if isinstance(src_bb, ResNet):
        # stem: giu nguyen
        _copy_sliced(dst_bb.conv1, src_bb.conv1,
                     torch.arange(src_bb.conv1.in_channels),
                     torch.arange(src_bb.conv1.out_channels))
        for s_stage, d_stage in zip(
            (src_bb.layer1, src_bb.layer2, src_bb.layer3, src_bb.layer4),
            (dst_bb.layer1, dst_bb.layer2, dst_bb.layer3, dst_bb.layer4),
        ):
            for s_blk, d_blk in zip(s_stage, d_stage):
                in_full = torch.arange(s_blk.conv1.in_channels)
                if isinstance(s_blk, BasicBlock):
                    mid = _kept_idx(s_blk.conv1)
                    out_full = torch.arange(s_blk.conv2.out_channels)
                    _copy_sliced(d_blk.conv1, s_blk.conv1, in_full, mid)
                    _copy_sliced(d_blk.conv2, s_blk.conv2, mid, out_full)
                else:  # BottleNeck
                    mid1 = _kept_idx(s_blk.conv1)
                    mid2 = _kept_idx(s_blk.conv2)
                    out_full = torch.arange(s_blk.conv3.out_channels)
                    _copy_sliced(d_blk.conv1, s_blk.conv1, in_full, mid1)
                    _copy_sliced(d_blk.conv2, s_blk.conv2, mid1, mid2)
                    _copy_sliced(d_blk.conv3, s_blk.conv3, mid2, out_full)
                if isinstance(s_blk.downsample, PrunableConv):
                    _copy_sliced(d_blk.downsample, s_blk.downsample,
                                 torch.arange(s_blk.downsample.in_channels),
                                 torch.arange(s_blk.downsample.out_channels))
        # backbone output = full (output block khong bi prune)
        last_blk = src_bb.layer4[-1]
        out_ch = (last_blk.conv2.out_channels if isinstance(last_blk, BasicBlock)
                  else last_blk.conv3.out_channels)
        return torch.arange(out_ch)

    elif isinstance(src_bb, VGG):
        prev = torch.arange(3)
        s_convs = [m for m in src_bb.features if isinstance(m, PrunableConv)]
        d_convs = [m for m in dst_bb.features if isinstance(m, PrunableConv)]
        for s_c, d_c in zip(s_convs, d_convs):
            out_idx = _kept_idx(s_c)
            _copy_sliced(d_c, s_c, prev, out_idx)
            prev = out_idx
        return prev
    else:
        raise TypeError(f"Unknown backbone type: {type(src_bb)}")


def _surgery_neck(src_neck, dst_neck, bb_out_idx):
    prev = bb_out_idx
    s_convs = [m for m in src_neck.convs if isinstance(m, PrunableConv)]
    d_convs = [m for m in dst_neck.convs if isinstance(m, PrunableConv)]
    for s_c, d_c in zip(s_convs, d_convs):
        out_idx = _kept_idx(s_c)
        _copy_sliced(d_c, s_c, prev, out_idx)
        prev = out_idx
    return prev


def _surgery_head(src_head, dst_head, in_idx):
    in_idx = in_idx.to(src_head.detect.weight.device)
    dst_head.detect.weight.data.copy_(src_head.detect.weight.data[:, in_idx])
    if src_head.detect.bias is not None:
        dst_head.detect.bias.data.copy_(src_head.detect.bias.data)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def convert_to_lean(model: YoloModel, save_path=None):
    """Tao lean YoloModel tu masked model. Tra ve (lean_model, config)."""
    model.eval()
    device = next(model.parameters()).device

    is_resnet = isinstance(model.backbone, ResNet)
    bb_cfg = _extract_resnet_cfg(model.backbone) if is_resnet else _extract_vgg_cfg(model.backbone)
    neck_w = _extract_neck_cfg(model.neck)

    config = {
        "backbone": model.backbone_name,
        "input_size": model.input_size,
        "num_classes": model.num_classes,
        "backbone_cfg": bb_cfg,
        "neck_widths": neck_w,
    }

    lean = build_lean_from_config(config).to(device)
    lean.eval()

    with torch.no_grad():
        bb_out_idx = _surgery_backbone(model.backbone, lean.backbone)
        neck_out_idx = _surgery_neck(model.neck, lean.neck, bb_out_idx)
        _surgery_head(model.head, lean.head, neck_out_idx)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(lean.state_dict(), save_path)
        with open(save_path.replace(".pth", ".json"), "w") as f:
            json.dump(config, f, indent=2)

    return lean, config


def build_lean_from_config(config):
    return YoloModel(
        input_size=config["input_size"],
        backbone=config["backbone"],
        num_classes=config["num_classes"],
        backbone_cfg=config["backbone_cfg"],
        neck_widths=config["neck_widths"],
    )
