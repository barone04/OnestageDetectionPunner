"""
YoloLoss (standalone) — theo cong thuc YOLOv1 reference.

Thanh phan: obj (MSE, target = IoU pred-gt), noobj (MSE, target 0),
box (MSE, chi cell co object, x lambda_coord), cls (CrossEntropy).
labels: list per-image tensor [cls, xc, yc, w, h] (chuan hoa [0,1]); cell trong => [-1,0,0,0,0].
"""
import torch
import torch.nn as nn


def set_grid(grid_size):
    grid_y, grid_x = torch.meshgrid(
        (torch.arange(grid_size), torch.arange(grid_size)), indexing="ij"
    )
    return grid_x, grid_y


class YoloLoss:
    def __init__(self, grid_size, num_classes, lambda_coord=5.0, lambda_noobj=0.5,
                 label_smoothing=0.0):
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.num_attributes = 1 + 4 + 1   # obj + box(4) + cls_id
        self.obj_loss_func = nn.MSELoss(reduction="none")
        self.box_loss_func = nn.MSELoss(reduction="none")
        self.cls_loss_func = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)
        gx, gy = set_grid(grid_size)
        self.grid_x = gx.contiguous().view((1, -1))
        self.grid_y = gy.contiguous().view((1, -1))

    def __call__(self, predictions, labels):
        self.device = predictions.device
        self.batch_size = predictions.shape[0]
        targets = self.build_batch_target(labels).to(self.device)

        with torch.no_grad():
            iou_pred_gt = self.calculate_iou(predictions[..., 1:5], targets[..., 1:5])

        pred_obj = predictions[..., 0]
        pred_box = predictions[..., 1:5]
        pred_cls = predictions[..., 5:].permute(0, 2, 1)

        target_obj = (targets[..., 0] == 1).float()
        target_noobj = (targets[..., 0] == 0).float()
        target_box = targets[..., 1:5]
        target_cls = targets[..., 5].long()

        obj_loss = (self.obj_loss_func(pred_obj, iou_pred_gt) * target_obj).sum() / self.batch_size
        noobj_loss = (self.obj_loss_func(pred_obj, target_obj * 0) * target_noobj).sum() / self.batch_size
        box_loss = (self.box_loss_func(pred_box, target_box).sum(dim=-1) * target_obj).sum() / self.batch_size
        cls_loss = (self.cls_loss_func(pred_cls, target_cls) * target_obj).sum() / self.batch_size

        total = obj_loss + self.lambda_noobj * noobj_loss + self.lambda_coord * box_loss + cls_loss
        return {"loss": total, "obj": obj_loss, "noobj": noobj_loss, "box": box_loss, "cls": cls_loss}

    def build_target(self, label):
        target = torch.zeros((self.grid_size, self.grid_size, self.num_attributes), dtype=torch.float32)
        if -1 in label[:, 0]:
            return target
        for item in label:
            cls_id = item[0].long()
            gi = (item[1] * self.grid_size).long()
            gj = (item[2] * self.grid_size).long()
            gi = gi.clamp(0, self.grid_size - 1)
            gj = gj.clamp(0, self.grid_size - 1)
            tx = item[1] * self.grid_size - gi
            ty = item[2] * self.grid_size - gj
            target[gj, gi, 0] = 1.0
            target[gj, gi, 1:5] = torch.tensor([tx, ty, item[3], item[4]])
            target[gj, gi, 5] = cls_id
        return target

    def build_batch_target(self, labels):
        bt = torch.stack([self.build_target(l) for l in labels], dim=0)
        return bt.view(self.batch_size, -1, self.num_attributes)

    def calculate_iou(self, pred_box, target_box):
        p = self._cxcywh_to_xyxy(pred_box)
        t = self._cxcywh_to_xyxy(target_box)
        x1 = torch.max(p[..., 0], t[..., 0])
        y1 = torch.max(p[..., 1], t[..., 1])
        x2 = torch.min(p[..., 2], t[..., 2])
        y2 = torch.min(p[..., 3], t[..., 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        union = (abs(pred_box[..., 2] * pred_box[..., 3])
                 + abs(target_box[..., 2] * target_box[..., 3]) - inter)
        iou = torch.zeros_like(inter)
        pos = inter > 0
        iou[pos] = inter[pos] / union[pos].clamp(min=1e-9)
        return iou

    def _cxcywh_to_xyxy(self, boxes):
        xc = (boxes[..., 0] + self.grid_x.to(self.device)) / self.grid_size
        yc = (boxes[..., 1] + self.grid_y.to(self.device)) / self.grid_size
        x1 = xc - boxes[..., 2] / 2
        y1 = yc - boxes[..., 3] / 2
        x2 = xc + boxes[..., 2] / 2
        y2 = yc + boxes[..., 3] / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)
