"""
mAP evaluation (standalone) cho YOLOv1 — khong phu thuoc pycocotools.

- Decode qua model.infer() -> (B, S*S, 6) = [score, xc, yc, w, h, label] (normalized).
- Loc theo conf, NMS theo tung lop, match GT theo IoU.
- AP all-point interpolation (VOC2010/COCO style).
- Tra mAP@0.5 va mAP@0.5:0.95 (trung binh 10 nguong IoU).
"""
import torch
from torchvision.ops import nms


def cxcywh_to_xyxy(b):
    x1 = b[..., 0] - b[..., 2] / 2
    y1 = b[..., 1] - b[..., 3] / 2
    x2 = b[..., 0] + b[..., 2] / 2
    y2 = b[..., 1] + b[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(a, b):
    """a (N,4), b (M,4) xyxy -> IoU (N,M)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _voc_ap(recall, precision):
    """All-point interpolation AP."""
    mrec = torch.cat([torch.zeros(1), recall, torch.ones(1)])
    mpre = torch.cat([torch.zeros(1), precision, torch.zeros(1)])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])
    idx = (mrec[1:] != mrec[:-1]).nonzero().squeeze(1)
    return ((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum().item()


@torch.no_grad()
def evaluate_map(model, loader, device, num_classes,
                 conf_thresh=0.001, nms_thresh=0.5, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]

    model.eval()
    dets = {c: [] for c in range(num_classes)}     # (img_id, score, box_xyxy)
    gts = {c: {} for c in range(num_classes)}       # img_id -> (M,4)
    n_gt = {c: 0 for c in range(num_classes)}
    img_id = 0

    total = max(len(loader), 1)
    print_every = max(total // 10, 1)
    print(f"  [mAP] collecting predictions from {total} batches", flush=True)
    for batch_index, (names, imgs, labels, shapes) in enumerate(loader):
        out = model.infer(imgs.to(device)).cpu()    # (B, SS, 6)
        for b in range(out.shape[0]):
            pred = out[b]
            score = pred[:, 0]
            box = cxcywh_to_xyxy(pred[:, 1:5])
            label = pred[:, 5].long()
            keep = score > conf_thresh
            score, box, label = score[keep], box[keep], label[keep]
            for c in range(num_classes):
                cm = label == c
                if cm.any():
                    bc, sc = box[cm], score[cm]
                    k = nms(bc, sc, nms_thresh)
                    for j in k.tolist():
                        dets[c].append((img_id, sc[j].item(), bc[j]))

            lab = labels[b]
            lab = lab[lab[:, 0] >= 0]
            if lab.numel() > 0:
                gb = cxcywh_to_xyxy(lab[:, 1:5])
                gc = lab[:, 0].long()
                for c in range(num_classes):
                    m = gc == c
                    if m.any():
                        gts[c][img_id] = gb[m]
                        n_gt[c] += int(m.sum())
            img_id += 1
        if batch_index % print_every == 0 or batch_index + 1 == total:
            print(f"  [mAP {batch_index + 1}/{total}]", flush=True)

    map_per_iou = {}
    print("  [mAP] matching detections across IoU thresholds", flush=True)
    for t in iou_thresholds:
        aps = []
        for c in range(num_classes):
            if n_gt[c] == 0:
                continue
            d = sorted(dets[c], key=lambda x: -x[1])
            matched = {k: torch.zeros(v.shape[0], dtype=torch.bool) for k, v in gts[c].items()}
            tp = torch.zeros(len(d))
            fp = torch.zeros(len(d))
            for i, (iid, sc, bx) in enumerate(d):
                g = gts[c].get(iid)
                if g is None or g.shape[0] == 0:
                    fp[i] = 1
                    continue
                ious = box_iou(bx[None], g)[0]
                best = int(ious.argmax())
                if ious[best] >= t and not matched[iid][best]:
                    tp[i] = 1
                    matched[iid][best] = True
                else:
                    fp[i] = 1
            tp_c, fp_c = tp.cumsum(0), fp.cumsum(0)
            recall = tp_c / (n_gt[c] + 1e-9)
            precision = tp_c / (tp_c + fp_c + 1e-9)
            aps.append(_voc_ap(recall, precision))
        map_per_iou[t] = sum(aps) / len(aps) if aps else 0.0

    map5095 = sum(map_per_iou.values()) / len(map_per_iou) if map_per_iou else 0.0
    return {
        "mAP@0.5": map_per_iou.get(0.5, 0.0),
        "mAP@0.75": map_per_iou.get(0.75, 0.0),
        "mAP@0.9": map_per_iou.get(0.9, 0.0),
        "mAP@0.5:0.95": map5095,
    }
