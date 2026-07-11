"""
YoloModel (standalone, prunable) — theo kien truc YOLOv1 reference:
  backbone -> neck (ConvBlock) -> head (1x1 conv) -> grid SxS, 1 box/cell.

Forward (giong reference):
  - train: tra (B, S*S, 5 + C) voi obj,box da sigmoid (loss tinh ben ngoai = YoloLoss).
  - eval : tra (B, S*S, 6) = [score, xc, yc, w, h, label].

Cac module CO THE PRUNE: backbone convs + neck convs (PrunableConv).
Head 1x1 KHONG prune output.
"""
import torch
import torch.nn as nn

from .backbone import build_backbone
from .neck import ConvBlock
from .head import YoloHead


def set_grid(grid_size):
    grid_y, grid_x = torch.meshgrid(
        (torch.arange(grid_size), torch.arange(grid_size)), indexing="ij"
    )
    return grid_x, grid_y


class YoloModel(nn.Module):
    def __init__(self, input_size=448, backbone="resnet18", num_classes=20,
                 neck_out=512, backbone_cfg=None, neck_widths=None,
                 pretrained_backbone=False):
        super().__init__()
        self.stride = 32
        self.input_size = input_size
        self.grid_size = input_size // self.stride
        self.num_classes = num_classes
        self.backbone_name = backbone
        self._neck_out = neck_out
        self._backbone_cfg = backbone_cfg
        self._neck_widths = neck_widths

        self.backbone, feat_dims = build_backbone(backbone, backbone_cfg=backbone_cfg,
                                                  pretrained=pretrained_backbone)
        self.neck = ConvBlock(in_channels=feat_dims, out_channels=neck_out, widths=neck_widths)
        self.head = YoloHead(in_channels=self.neck.out_channels, num_classes=num_classes)

        grid_x, grid_y = set_grid(self.grid_size)
        self.register_buffer("grid_x", grid_x.contiguous().view((1, -1)).float(), persistent=False)
        self.register_buffer("grid_y", grid_y.contiguous().view((1, -1)).float(), persistent=False)

    def forward(self, x):
        """Luon tra RAW prediction (B, S*S, 5+C) voi obj,box da sigmoid -> dung cho YoloLoss."""
        out = self.backbone(x)
        out = self.neck(out)
        out = self.head(out)
        out = out.permute(0, 2, 3, 1).contiguous().flatten(1, 2)   # (B, S*S, 5+C)

        pred_obj = torch.sigmoid(out[..., [0]])
        pred_box = torch.sigmoid(out[..., 1:5])
        pred_cls = out[..., 5:]
        return torch.cat((pred_obj, pred_box, pred_cls), dim=-1)

    @torch.no_grad()
    def infer(self, x):
        """Decode -> (B, S*S, 6) = [score, xc, yc, w, h, label] (chua NMS)."""
        preds = self.forward(x)
        pred_obj = preds[..., [0]]
        pred_box = self._decode_box(preds[..., 1:5])
        pred_cls = preds[..., 5:]
        pred_score = pred_obj * torch.softmax(pred_cls, dim=-1)
        pred_score, pred_label = pred_score.max(dim=-1)
        return torch.cat((pred_score.unsqueeze(-1), pred_box, pred_label.unsqueeze(-1).float()), dim=-1)

    def _decode_box(self, pred_box):
        xc = (pred_box[..., 0] + self.grid_x) / self.grid_size
        yc = (pred_box[..., 1] + self.grid_y) / self.grid_size
        w = pred_box[..., 2]
        h = pred_box[..., 3]
        return torch.stack((xc, yc, w, h), dim=-1)

    # ----- prunable modules -----
    def get_prunable_layers(self, pruning_type="unstructured"):
        return (self.backbone.get_prunable_layers(pruning_type)
                + self.neck.get_prunable_layers(pruning_type))

    def get_backbone_prunable_layers(self, pruning_type="unstructured"):
        return self.backbone.get_prunable_layers(pruning_type)

    def get_neck_prunable_layers(self, pruning_type="unstructured"):
        return self.neck.get_prunable_layers(pruning_type)

    def config(self):
        cfg = {
            "backbone": self.backbone_name,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "neck_out": self._neck_out,
            "backbone_cfg": self._backbone_cfg,
            "neck_widths": self._neck_widths,
        }
        if getattr(self, "_variant", None) == "norton":
            cfg.update({
                "variant": "norton",
                "norton_rank": int(self._norton_rank),
                "norton_scope": self._norton_scope,
            })
        return cfg


def build_model(cfg):
    """Dung YoloModel tu dict config (dung cho reload / surgery rebuild)."""
    if cfg.get("variant") == "norton":
        from .norton import build_norton_model_from_config
        return build_norton_model_from_config(cfg)
    return YoloModel(
        input_size=cfg.get("input_size", 448),
        backbone=cfg.get("backbone", "resnet18"),
        num_classes=cfg.get("num_classes", 1),
        neck_out=cfg.get("neck_out", 512),
        backbone_cfg=cfg.get("backbone_cfg", None),
        neck_widths=cfg.get("neck_widths", None),
    )
